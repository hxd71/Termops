from __future__ import annotations

import asyncio
import hmac
import time
from typing import Any, Literal

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse, PlainTextResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import __version__
from .config import Settings
from .engine import OpsEngine
from .metrics import registry as metrics_registry
from .models import ApprovalDecision, FileAttachment, TargetRef, TaskKind
from .web import build_web_router, static_dir


class _TokenBucket:
    """Per-key token bucket rate limiter (in-memory, thread-unsafe by design:
    it only runs inside the event loop)."""

    def __init__(self, rate: float, burst: int) -> None:
        self.rate = rate
        self.burst = burst
        self._buckets: dict[str, list[float]] = {}

    def allow(self, key: str) -> bool:
        now = time.monotonic()
        if len(self._buckets) > 4096:  # bound memory: evict keys idle for >10 min
            self._buckets = {k: v for k, v in self._buckets.items() if now - v[1] < 600}
        tokens, last = self._buckets.get(key, [float(self.burst), now])
        tokens = min(float(self.burst), tokens + (now - last) * self.rate)
        if tokens < 1.0:
            self._buckets[key] = [tokens, now]
            return False
        self._buckets[key] = [tokens - 1.0, now]
        return True


class AnalyzeRequest(BaseModel):
    text: str = Field(min_length=1, max_length=200_000)
    source: str = Field(default="stdin", max_length=64)
    language: str = Field(default="", max_length=64)
    command: str = Field(default="", max_length=2000)
    cwd: str = Field(default="", max_length=2000)
    exit_code: int | None = Field(default=None, ge=-4096, le=4096)
    files: list[FileAttachment] = Field(default_factory=list)
    history_task_ids: list[str] = Field(default_factory=list, max_length=10)


class RunRequest(BaseModel):
    command: list[str] = Field(min_length=1)
    cwd: str = Field(default="", max_length=2000)
    language: str = Field(default="", max_length=64)
    timeout: int = Field(default=120, ge=1, le=3600)


class KnowledgeRecordRequest(BaseModel):
    kind: str = Field(default="note", max_length=64)
    title: str = Field(min_length=1, max_length=256)
    content: dict[str, Any] = Field(default_factory=dict)


class KnowledgeSearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    limit: int = Field(default=5, ge=1, le=50)


class CorrectionRequest(BaseModel):
    sample_text: str = Field(min_length=1, max_length=2000)
    code: str = Field(min_length=1, max_length=64)
    severity: Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"] = "MEDIUM"
    meaning: str = Field(min_length=1, max_length=500)
    remediation: str | None = Field(default=None, max_length=1000)


class ProbeRequest(BaseModel):
    cwd: str = Field(default="", max_length=2000)
    language: str = Field(default="", max_length=64)


def create_app(settings: Settings, engine: OpsEngine) -> FastAPI:
    app = FastAPI(
        title="Local Error Analysis Agent",
        version=__version__,
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )
    app.state.settings = settings
    app.state.engine = engine
    app.mount("/static", StaticFiles(directory=static_dir()), name="static")

    # Mutation endpoints get a per-client token bucket so a script cannot flood
    # the engine with LLM-backed analyses. The concurrency Semaphore limits
    # *parallelism*; this limits *request rate* — a different axis of abuse.
    submit_limiter = _TokenBucket(rate=2.0, burst=10)
    mutation_prefixes = ("/v1/tasks", "/v1/knowledge", "/v1/corrections", "/v1/actions")

    @app.middleware("http")
    async def bind_engine_loop(request: Request, call_next: Any) -> Any:
        # Bind the serving loop on every request so off-thread submissions
        # (engine.submit_* / decide_action) can schedule coroutines back onto it.
        engine.bind_loop(asyncio.get_running_loop())
        if request.method in {"POST", "DELETE"} and request.url.path.startswith(mutation_prefixes):
            client = request.client.host if request.client else "unknown"
            metrics_registry.inc("termops_http_requests_total")
            if not submit_limiter.allow(client):
                metrics_registry.inc("termops_http_rate_limited_total")
                return JSONResponse({"detail": "rate limit exceeded; slow down"}, status_code=429)
        return await call_next(request)

    def require_operator_token(x_operator_token: str | None = Header(default=None)) -> None:
        if not x_operator_token or not hmac.compare_digest(x_operator_token, engine.operator_token):
            raise HTTPException(status_code=401, detail="valid local operator token required")

    @app.exception_handler(KeyError)
    async def key_error_handler(_request: Request, exc: KeyError) -> JSONResponse:
        return JSONResponse({"detail": str(exc)}, status_code=404)

    @app.exception_handler(ValueError)
    async def value_error_handler(_request: Request, exc: ValueError) -> JSONResponse:
        return JSONResponse({"detail": str(exc)}, status_code=422)

    @app.get("/healthz")
    async def healthz() -> dict[str, Any]:
        caps = await asyncio.to_thread(engine.capabilities)
        return {
            "ok": True,
            "profile": settings.profile,
            "agent": "Local Error Analysis Agent",
            "description": "MAPE-K powered terminal/code error analysis with LLM attribution",
            "llm": caps.get("llm", {}),
            "mape_k": caps.get("orchestration", {}),
        }

    @app.get("/metrics", dependencies=[Depends(require_operator_token)])
    async def metrics() -> PlainTextResponse:
        """Prometheus-format metrics: task throughput, durations, LLM usage, rate limiting."""
        return PlainTextResponse(metrics_registry.render_prometheus())

    @app.get("/", include_in_schema=False)
    async def root() -> RedirectResponse:
        return RedirectResponse("/ui/", status_code=307)

    @app.get("/v1/capabilities", dependencies=[Depends(require_operator_token)])
    async def capabilities() -> dict[str, Any]:
        return await asyncio.to_thread(engine.capabilities)

    @app.post("/v1/tasks/analyze", dependencies=[Depends(require_operator_token)], status_code=202)
    async def analyze(request: AnalyzeRequest) -> dict[str, Any]:
        task = await asyncio.to_thread(
            engine.submit_analysis,
            request.text,
            source=request.source,
            language=request.language,
            command=request.command,
            cwd=request.cwd,
            exit_code=request.exit_code,
            files=request.files,
            history_task_ids=request.history_task_ids,
        )
        return task.model_dump(mode="json")

    @app.post("/v1/tasks/run", dependencies=[Depends(require_operator_token)], status_code=202)
    async def run_command(request: RunRequest) -> dict[str, Any]:
        task = await asyncio.to_thread(
            engine.submit_run,
            request.command,
            cwd=request.cwd,
            language=request.language,
            timeout=request.timeout,
        )
        return task.model_dump(mode="json")

    @app.post("/v1/tasks/probe", dependencies=[Depends(require_operator_token)], status_code=202)
    async def probe(request: ProbeRequest) -> dict[str, Any]:
        task = await asyncio.to_thread(engine.submit_probe, cwd=request.cwd, language=request.language)
        return task.model_dump(mode="json")

    @app.post("/v1/knowledge", dependencies=[Depends(require_operator_token)], status_code=202)
    async def record_knowledge(request: KnowledgeRecordRequest) -> dict[str, Any]:
        task = await asyncio.to_thread(
            engine.store.create_task,
            TaskKind.KNOWLEDGE_RECORD,
            TargetRef(kind="workspace", name="knowledge"),
            {"knowledge": request.model_dump()},
        )
        return task.model_dump(mode="json")

    @app.get("/v1/tasks", dependencies=[Depends(require_operator_token)])
    async def list_tasks(limit: int = 100) -> list[dict[str, Any]]:
        tasks = await asyncio.to_thread(engine.store.list_tasks, min(max(limit, 1), 500))
        return [item.model_dump(mode="json") for item in tasks]

    @app.get("/v1/tasks/{task_id}", dependencies=[Depends(require_operator_token)])
    async def task_detail(task_id: str) -> dict[str, Any]:
        return await asyncio.to_thread(engine.task_detail, task_id)

    @app.get("/v1/tasks/{task_id}/events", dependencies=[Depends(require_operator_token)])
    async def task_events(task_id: str, after: int = 0) -> list[dict[str, Any]]:
        await asyncio.to_thread(engine.store.get_task, task_id)
        return await asyncio.to_thread(engine.store.list_events, task_id, after_seq=max(after, 0))

    @app.get("/v1/tasks/{task_id}/knowledge", dependencies=[Depends(require_operator_token)])
    async def task_knowledge(task_id: str) -> list[dict[str, Any]]:
        await asyncio.to_thread(engine.store.get_task, task_id)
        return await asyncio.to_thread(engine.store.list_knowledge, task_id)

    @app.post("/v1/tasks/{task_id}/cancel", dependencies=[Depends(require_operator_token)])
    async def cancel_task(task_id: str) -> dict[str, Any]:
        task = await asyncio.to_thread(engine.cancel_task, task_id)
        return task.model_dump(mode="json")

    @app.get("/v1/actions/{action_id}", dependencies=[Depends(require_operator_token)])
    async def get_action(action_id: str) -> dict[str, Any]:
        action = await asyncio.to_thread(engine.store.get_action, action_id)
        return action.model_dump(mode="json")

    @app.post("/v1/actions/{action_id}/decision", dependencies=[Depends(require_operator_token)])
    async def decide_action(action_id: str, decision: ApprovalDecision) -> dict[str, Any]:
        action = await asyncio.to_thread(engine.decide_action, action_id, decision)
        return action.model_dump(mode="json")

    @app.get("/v1/events", dependencies=[Depends(require_operator_token)])
    async def audit_events(limit: int = 200) -> list[dict[str, Any]]:
        return await asyncio.to_thread(engine.store.list_events, limit=min(max(limit, 1), 500))

    @app.get("/v1/knowledge", dependencies=[Depends(require_operator_token)])
    async def knowledge(limit: int = 200) -> list[dict[str, Any]]:
        return await asyncio.to_thread(engine.store.list_knowledge, limit=min(max(limit, 1), 500))

    @app.post("/v1/knowledge/search", dependencies=[Depends(require_operator_token)])
    async def search_knowledge(request: KnowledgeSearchRequest) -> dict[str, Any]:
        results = await asyncio.to_thread(engine.store.search_knowledge, request.query, request.limit)
        return {
            "query": request.query,
            "count": len(results),
            "results": results,
            "fts_enabled": engine.store._fts_enabled,
        }

    @app.get("/v1/knowledge/stats", dependencies=[Depends(require_operator_token)])
    async def knowledge_stats() -> dict[str, Any]:
        return await asyncio.to_thread(engine.store.knowledge_stats)

    @app.post("/v1/corrections", dependencies=[Depends(require_operator_token)], status_code=201)
    async def record_correction(request: CorrectionRequest) -> dict[str, Any]:
        return await asyncio.to_thread(
            engine.record_correction,
            request.sample_text,
            request.code,
            request.severity,
            request.meaning,
            request.remediation,
        )

    @app.get("/v1/corrections", dependencies=[Depends(require_operator_token)])
    async def list_corrections(limit: int = 100) -> list[dict[str, Any]]:
        return await asyncio.to_thread(engine.store.list_corrections, min(max(limit, 1), 500))

    @app.delete("/v1/corrections/{fingerprint}", dependencies=[Depends(require_operator_token)])
    async def delete_correction(fingerprint: str) -> dict[str, Any]:
        deleted = await asyncio.to_thread(engine.store.delete_correction, fingerprint)
        engine.invalidate_corrections()
        return {"deleted": deleted}

    @app.post("/v1/web/login-code", dependencies=[Depends(require_operator_token)])
    async def create_login_code() -> dict[str, Any]:
        from .security import new_token

        code = new_token()
        await asyncio.to_thread(engine.store.create_login_code, code)
        return {"url": f"http://{settings.web_host}:{settings.web_port}/login?code={code}", "expires_in": 120}

    app.include_router(build_web_router(settings, engine))
    return app