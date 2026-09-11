"""MAPE-K analysis engine for terminal and code errors.

The engine treats every task as a loop through Monitor, Analyze, Plan, Execute,
Observe and Knowledge. State transitions are explicit, persisted, and auditable.
Only actions that have passed human approval are executed.
"""

from __future__ import annotations

import asyncio
import logging
import os
import shlex
import subprocess
import time
from collections.abc import Callable, Coroutine
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path
from typing import Any

from langchain_core.runnables import RunnableConfig

from .config import Settings
from .diagnostics import classify_error
from .graph import build_analysis_graph, build_execution_graph
from .llm_client import LLMClient
from .metrics import registry as metrics_registry
from .models import (
    TERMINAL_TASK_STATUSES,
    ActionStatus,
    AnalysisReport,
    AnalysisRequest,
    ApprovalDecision,
    EnvSnapshot,
    FileAttachment,
    LLMAttribution,
    MapePhase,
    Severity,
    TargetRef,
    Task,
    TaskKind,
    TaskStatus,
)
from .providers import EnvProbe
from .security import new_token
from .store import StateStore

# Cap concurrent MAPE-K task executions so a burst of submissions cannot exhaust
# the event loop or the LLM provider's rate limit. Bounded at the same level as
# the database worker pool to keep resource contention predictable.
MAX_CONCURRENT_TASKS = 8
DB_WORKERS = 8

log = logging.getLogger("termops.engine")


class OpsEngine:
    """Local error analysis engine with approval-gated action execution."""

    def __init__(
        self,
        settings: Settings,
        store: StateStore | None = None,
        probe: EnvProbe | None = None,
    ):
        self.settings = settings
        self.settings.ensure_directories()
        self.store = store or StateStore(settings.database_path)
        self.probe = probe or EnvProbe(env_allowlist=settings.env_allowlist)
        self._jobs: set[asyncio.Task[Any]] = set()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._task_semaphore = asyncio.Semaphore(MAX_CONCURRENT_TASKS)
        self._db_executor = ThreadPoolExecutor(max_workers=DB_WORKERS, thread_name_prefix="termops-db")
        self._analysis_graph: Any | None = None
        self._execution_graph: Any | None = None
        self._corrections_cache: dict[str, dict[str, Any]] | None = None
        self._corrections_loaded_at = 0.0
        self.llm = LLMClient(settings.llm)
        self.operator_token = self._load_or_create_operator_token()

    @property
    def analysis_graph(self) -> Any:
        if self._analysis_graph is None:
            self._analysis_graph = build_analysis_graph()
        return self._analysis_graph

    @property
    def execution_graph(self) -> Any:
        if self._execution_graph is None:
            self._execution_graph = build_execution_graph()
        return self._execution_graph

    def _load_or_create_operator_token(self) -> str:
        path = self.settings.operator_token_path
        if path.exists():
            return path.read_text(encoding="utf-8").strip()
        token = new_token()
        path.write_text(token + "\n", encoding="utf-8")
        if self.settings.profile == "live" and Path(path).drive == "":
            path.chmod(0o600)
        return token

    async def start(self) -> None:
        self._loop = asyncio.get_running_loop()
        recoverable = {
            TaskStatus.EXECUTING,
            TaskStatus.VERIFYING,
            TaskStatus.QUEUED,
        }
        for task in self.store.list_tasks(limit=None, statuses=recoverable):
            if task.status in {TaskStatus.EXECUTING, TaskStatus.VERIFYING}:
                self.store.update_task(task.id, TaskStatus.RECONCILING, force=True)
                self._spawn(self._reconcile_task(task.id))
            elif task.status == TaskStatus.QUEUED:
                self._spawn(self.process_task(task.id))

    async def stop(self) -> None:
        for job in list(self._jobs):
            job.cancel()
        if self._jobs:
            await asyncio.gather(*self._jobs, return_exceptions=True)
        self._db_executor.shutdown(wait=True, cancel_futures=True)

    def bind_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """Bind the owning event loop so off-thread submissions can schedule onto it."""
        current = self._loop
        if current is None or current.is_closed():
            self._loop = loop

    def _spawn(self, coroutine: Coroutine[Any, Any, Any]) -> None:
        """Schedule a tracked task on the owning event loop, from any thread."""
        loop = self._loop
        if loop is None or loop.is_closed():
            loop = asyncio.get_running_loop()
        if not loop.is_running():
            raise RuntimeError("cannot spawn task: no running event loop")
        loop.call_soon_threadsafe(self._track_task, self._gated(coroutine))

    def _track_task(self, coroutine: Coroutine[Any, Any, Any]) -> None:
        job = asyncio.get_running_loop().create_task(coroutine)
        self._jobs.add(job)
        metrics_registry.set_gauge("termops_queue_depth", float(len(self._jobs)))
        job.add_done_callback(self._on_job_done)

    def _on_job_done(self, job: asyncio.Task[Any]) -> None:
        self._jobs.discard(job)
        metrics_registry.set_gauge("termops_queue_depth", float(len(self._jobs)))

    async def _gated(self, coroutine: Coroutine[Any, Any, Any]) -> None:
        async with self._task_semaphore:
            await coroutine

    async def _db(self, func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """Run a blocking store call in the DB worker pool, off the event loop."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._db_executor, partial(func, *args, **kwargs))

    @staticmethod
    def _split_command(command: str) -> list[str]:
        parts = shlex.split(command, posix=False)
        return [part[1:-1] if len(part) >= 2 and part[0] == part[-1] == '"' else part for part in parts]

    def _mark_phase(self, task_id: str, phase: MapePhase, detail: str = "") -> None:
        self.store.update_task_phase(task_id, phase)
        self.store.append_event("task.phase", {"phase": phase.value, "detail": detail}, task_id)

    def task_detail(self, task_id: str) -> dict[str, Any]:
        task = self.store.get_task(task_id)
        return {
            "task": task.model_dump(mode="json"),
            "observations": [item.model_dump(mode="json") for item in self.store.list_observations(task_id)],
            "findings": [item.model_dump(mode="json") for item in self.store.list_findings(task_id)],
            "actions": [item.model_dump(mode="json") for item in self.store.list_actions(task_id)],
            "knowledge": self.store.list_knowledge(task_id),
            "events": self.store.list_events(task_id, limit=200),
        }

    def capabilities(self) -> dict[str, Any]:
        return {
            "profile": self.settings.profile,
            "host": self.probe.capabilities(),
            "llm": {
                "provider": self.llm.config.provider.value,
                "enabled": self.llm.enabled,
                "model": self.llm.config.model if self.llm.enabled else None,
                "reason": "llm attribution active" if self.llm.enabled else "deterministic production core",
            },
            "orchestration": {"mape_k": True, "framework": "langgraph"},
            "event_chain_valid": self.store.verify_event_chain(),
        }

    # ------------------------------------------------------------------
    # Operator corrections (feedback loop)
    # ------------------------------------------------------------------

    _CORRECTIONS_TTL = 30.0

    def get_corrections(self) -> dict[str, dict[str, Any]]:
        """Return the corrections map, refreshing from the store at most every TTL."""
        now = time.monotonic()
        if self._corrections_cache is None or (now - self._corrections_loaded_at) > self._CORRECTIONS_TTL:
            self._corrections_cache = self.store.get_corrections()
            self._corrections_loaded_at = now
        return self._corrections_cache

    def invalidate_corrections(self) -> None:
        self._corrections_cache = None

    def record_correction(
        self,
        sample_text: str,
        code: str,
        severity: str,
        meaning: str,
        remediation: str | None = None,
    ) -> dict[str, Any]:
        """Persist an operator correction and make it effective immediately."""
        from .diagnostics import normalize_error_text

        fingerprint = normalize_error_text(sample_text)
        result = self.store.upsert_correction(
            fingerprint=fingerprint,
            sample=sample_text,
            code=code,
            severity=severity,
            meaning=meaning,
            remediation=remediation,
            created_by=self.settings.operator_name,
        )
        self.invalidate_corrections()
        return result

    def classify(self, text: str, exit_code: int | None = None) -> dict[str, Any]:
        """Classify error text with operator corrections applied."""
        result = classify_error(text, exit_code, corrections=self.get_corrections())
        for finding in result.get("findings", []):
            if finding.get("source") == "correction" and finding.get("fingerprint"):
                self.store.increment_correction_hits(finding["fingerprint"])
        return result

    # ------------------------------------------------------------------
    # Public submission API
    # ------------------------------------------------------------------

    def submit_analysis(
        self,
        text: str,
        *,
        source: str = "stdin",
        language: str = "",
        command: str = "",
        cwd: str = "",
        exit_code: int | None = None,
        files: list[FileAttachment] | None = None,
        history_task_ids: list[str] | None = None,
    ) -> Task:
        request = AnalysisRequest(
            text=text,
            source=source,
            language=language,
            command=command,
            cwd=cwd,
            exit_code=exit_code,
            files=files or [],
            history_task_ids=history_task_ids or [],
        )
        task = self.store.create_task(
            TaskKind.ANALYZE,
            TargetRef(kind="terminal" if command else "workspace", name=request.source),
            request.model_dump(mode="json"),
        )
        self._spawn(self.process_task(task.id))
        return task

    def submit_run(
        self,
        command: list[str],
        *,
        cwd: str = "",
        language: str = "",
        timeout: int = 120,
    ) -> Task:
        request = AnalysisRequest(
            text="",
            source="cli_run",
            language=language,
            command=subprocess.list2cmdline(command) if os.name == "nt" else shlex.join(command),
            cwd=cwd,
            exit_code=None,
        )
        task = self.store.create_task(
            TaskKind.ANALYZE,
            TargetRef(kind="command", name=request.command),
            {
                **request.model_dump(mode="json"),
                "run": {"command": command, "cwd": cwd, "timeout": timeout},
            },
        )
        self._spawn(self.process_task(task.id))
        return task

    def submit_probe(self, *, cwd: str = "", language: str = "") -> Task:
        request = AnalysisRequest(
            text="environment probe",
            source="probe",
            language=language,
            cwd=cwd,
        )
        task = self.store.create_task(
            TaskKind.PROBE,
            TargetRef(kind="workspace", name="local"),
            request.model_dump(mode="json"),
        )
        self._spawn(self.process_task(task.id))
        return task

    def cancel_task(self, task_id: str) -> Task:
        task = self.store.get_task(task_id)
        if task.status not in {TaskStatus.QUEUED, TaskStatus.RUNNING, TaskStatus.WAITING_APPROVAL}:
            raise ValueError(f"task cannot be cancelled from {task.status.value}")
        for action in self.store.list_actions(task_id):
            if action.status == ActionStatus.PENDING:
                self.store.update_action(
                    action.id,
                    ActionStatus.REJECTED,
                    {"ok": False, "reason": "task cancelled by operator"},
                )
        return self.store.update_task(task_id, TaskStatus.CANCELLED)

    # ------------------------------------------------------------------
    # MAPE-K state loop
    # ------------------------------------------------------------------

    async def process_task(self, task_id: str) -> None:
        task = await self._db(self.store.get_task, task_id)
        if task.status != TaskStatus.QUEUED:
            return
        await self._db(self.store.update_task, task_id, TaskStatus.RUNNING, phase=MapePhase.MONITOR)
        metrics_registry.inc("termops_tasks_started_total")
        started = time.perf_counter()
        log.info("task started", extra={"task_id": task_id, "kind": task.kind.value})
        try:
            if task.kind == TaskKind.ANALYZE:
                await self._run_analysis_graph(task_id)
            elif task.kind == TaskKind.VERIFY:
                await self._verify_task(task_id)
            elif task.kind == TaskKind.PROBE:
                await self._probe_task(task_id)
            elif task.kind == TaskKind.KNOWLEDGE_RECORD:
                await self._knowledge_task(task_id)
            else:
                raise ValueError(f"unsupported task kind: {task.kind.value}")
            metrics_registry.inc("termops_tasks_succeeded_total")
            log.info(
                "task succeeded",
                extra={"task_id": task_id, "duration_s": round(time.perf_counter() - started, 3)},
            )
        except Exception as exc:
            metrics_registry.inc("termops_tasks_failed_total")
            log.warning(
                "task failed",
                extra={"task_id": task_id, "error": str(exc)[:500]},
            )
            current = await self._db(self.store.get_task, task_id)
            if current.status not in TERMINAL_TASK_STATUSES:
                await self._db(
                    self.store.update_task,
                    task_id,
                    TaskStatus.FAILED,
                    error=str(exc),
                    phase=MapePhase.KNOWLEDGE,
                    force=True,
                )
        finally:
            metrics_registry.observe("termops_analysis_duration_seconds", time.perf_counter() - started)

    async def _run_analysis_graph(self, task_id: str) -> None:
        config: RunnableConfig = RunnableConfig(configurable={"engine": self})
        await self.analysis_graph.ainvoke({"task_id": task_id}, config=config)

    async def _run_execution_graph(self, task_id: str, action_id: str) -> None:
        config: RunnableConfig = RunnableConfig(configurable={"engine": self})
        await self.execution_graph.ainvoke({"task_id": task_id, "action_id": action_id}, config=config)

    async def _verify_task(self, task_id: str) -> None:
        """Run a verification command and analyze its output."""
        await self._db(self._mark_phase, task_id, MapePhase.MONITOR, "capturing verification input")
        request, _env = await self._monitor(task_id)
        task = await self._db(self.store.get_task, task_id)
        run = task.input.get("run")
        if not run:
            raise ValueError("VERIFY task is missing run payload")

        await self._db(self._mark_phase, task_id, MapePhase.EXECUTE, "running verification command")
        result = await self._run_command(run["command"], run.get("cwd"), run.get("timeout", 120))
        obs = await self._db(
            self.store.add_observation,
            task_id,
            "command_result",
            "verify",
            "ok" if result["returncode"] == 0 else "warning",
            f"Verification command exited with code {result['returncode']}.",
            result,
        )

        await self._db(self._mark_phase, task_id, MapePhase.OBSERVE, "analyzing verification output")
        verify_text = result["stdout"] + "\n" + result["stderr"]
        verification = await self._db(self.classify, verify_text, result["returncode"])
        for match in verification["findings"]:
            await self._db(
                self.store.add_finding,
                task_id,
                match["code"],
                Severity(str(match["severity"]).lower()),
                0.85,
                match["meaning"],
                f"Line {match['line']}: {match['text']}",
                [obs.id],
                match["remediation"],
            )

        await self._finish_analysis(
            task_id,
            {"codes": set(), "findings": []},
            request,
            summary=f"Verification command exited with code {result['returncode']}.",
        )

    async def _probe_task(self, task_id: str) -> None:
        await self._db(self._mark_phase, task_id, MapePhase.MONITOR, "capturing environment snapshot")
        task = await self._db(self.store.get_task, task_id)
        request = AnalysisRequest.model_validate(task.input)
        env = self.probe.snapshot(cwd=request.cwd, language=request.language)
        await self._db(
            self.store.add_observation,
            task_id,
            "env_snapshot",
            "EnvProbe",
            "ok",
            "Collected a safe view of the local execution environment.",
            env.model_dump(mode="json"),
        )
        report = await self._db(self._build_report, task_id, "Environment snapshot completed.")
        await self._db(
            self.store.update_task,
            task_id,
            TaskStatus.SUCCEEDED,
            phase=MapePhase.KNOWLEDGE,
            report=report,
        )

    async def _knowledge_task(self, task_id: str) -> None:
        await self._db(self._mark_phase, task_id, MapePhase.KNOWLEDGE, "recording explicit knowledge")
        task = await self._db(self.store.get_task, task_id)
        entry = task.input.get("knowledge", {})
        await self._db(
            self.store.add_knowledge,
            task_id,
            entry.get("kind", "note"),
            entry.get("title", "Untitled"),
            entry.get("content", {}),
        )
        report = await self._db(self._build_report, task_id, "Knowledge entry recorded.")
        await self._db(
            self.store.update_task,
            task_id,
            TaskStatus.SUCCEEDED,
            phase=MapePhase.KNOWLEDGE,
            report=report,
        )

    async def _monitor(self, task_id: str) -> tuple[AnalysisRequest, EnvSnapshot]:
        task = await self._db(self.store.get_task, task_id)
        request_data = dict(task.input)
        request_data.pop("run", None)
        request = AnalysisRequest.model_validate(request_data)

        # If this is a wrapped run, execute the command first.
        run = task.input.get("run")
        if run and task.kind == TaskKind.ANALYZE:
            result = await self._run_command(run["command"], run.get("cwd"), run.get("timeout", 120))
            request.text = "\n".join(part for part in [result["stdout"], result["stderr"]] if part)
            request.exit_code = result["returncode"]

        env = self.probe.snapshot(cwd=request.cwd, language=request.language)
        await self._db(
            self.store.add_observation,
            task_id,
            "analysis_input",
            request.source,
            "ok",
            "Captured the submitted error context.",
            {
                **request.model_dump(mode="json"),
                "env": env.model_dump(mode="json"),
            },
        )
        return request, env

    async def _finish_analysis(
        self,
        task_id: str,
        classification: dict[str, Any],
        request: AnalysisRequest,
        summary: str | None = None,
        llm_attribution: LLMAttribution | None = None,
        retrieved_knowledge_ids: list[str] | None = None,
    ) -> None:
        await self._db(self._mark_phase, task_id, MapePhase.KNOWLEDGE, "recording analysis outcome")
        findings = await self._db(self.store.list_findings, task_id)
        if summary is None:
            summary = (
                f"Analysis found {len(findings)} likely issue(s)."
                if findings
                else "Analysis completed with no actionable findings."
            )
        await self._db(
            self.store.add_knowledge,
            task_id,
            "analysis_summary",
            summary,
            {
                "input": request.model_dump(mode="json"),
                "codes": classification.get("codes", []),
                "findings": [item.model_dump(mode="json") for item in findings],
                "llm_attribution": llm_attribution.model_dump(mode="json") if llm_attribution else None,
                "retrieved_knowledge_ids": retrieved_knowledge_ids or [],
            },
        )
        report = await self._db(
            self._build_report,
            task_id,
            summary,
            llm_attribution=llm_attribution,
            retrieved_knowledge_ids=retrieved_knowledge_ids,
        )
        await self._db(
            self.store.update_task,
            task_id,
            TaskStatus.SUCCEEDED,
            phase=MapePhase.KNOWLEDGE,
            report=report,
        )

    async def _run_command(
        self, command: list[str] | str, cwd: str | None = None, timeout: int = 120
    ) -> dict[str, Any]:
        args = self._split_command(command) if isinstance(command, str) else list(command)
        if not args:
            raise ValueError("command is empty")
        resolved_cwd = cwd if cwd and Path(cwd).exists() else None
        command_line = subprocess.list2cmdline(args) if os.name == "nt" else shlex.join(args)
        try:
            process = await asyncio.to_thread(
                subprocess.run,
                args,
                capture_output=True,
                text=True,
                cwd=resolved_cwd,
                shell=False,
                timeout=timeout,
                check=False,
            )
        except FileNotFoundError:
            # Missing binary (e.g. a suggested command not on this PATH) is a
            # *result*, not a task-fatal error.
            return {
                "command": command_line,
                "cwd": resolved_cwd,
                "returncode": 127,
                "stdout": "",
                "stderr": f"command not found or not on PATH: {args[0]}",
            }
        except subprocess.TimeoutExpired as exc:
            def _tail(value: object) -> str:
                if isinstance(value, bytes):
                    return value.decode(errors="replace")[-8000:]
                return value[-8000:] if isinstance(value, str) else ""

            return {
                "command": command_line,
                "cwd": resolved_cwd,
                "returncode": 124,
                "stdout": _tail(exc.output),
                "stderr": (f"command timed out after {timeout}s\n" + _tail(exc.stderr))[-8000:],
            }
        return {
            "command": command_line,
            "cwd": resolved_cwd,
            "returncode": process.returncode,
            "stdout": process.stdout[-8000:],
            "stderr": process.stderr[-8000:],
        }

    def _build_report(
        self,
        task_id: str,
        summary: str,
        *,
        llm_attribution: LLMAttribution | None = None,
        retrieved_knowledge_ids: list[str] | None = None,
    ) -> dict[str, Any]:
        observations = self.store.list_observations(task_id)
        findings = self.store.list_findings(task_id)
        actions = self.store.list_actions(task_id)
        report = AnalysisReport(
            task_id=task_id,
            summary=summary,
            observation_count=len(observations),
            finding_count=len(findings),
            action_ids=[action.id for action in actions],
            suggested_next_steps=[finding.remediation for finding in findings if finding.remediation][:5],
            llm_attribution=llm_attribution,
            retrieved_knowledge_ids=retrieved_knowledge_ids or [],
        )
        return report.model_dump(mode="json")

    # ------------------------------------------------------------------
    # Approval and execution
    # ------------------------------------------------------------------

    def decide_action(self, action_id: str, decision: ApprovalDecision) -> Any:
        pending = self.store.get_action(action_id)
        task = self.store.get_task(pending.task_id)
        if task.status != TaskStatus.WAITING_APPROVAL:
            raise ValueError(f"task is not waiting for approval: {task.status.value}")
        try:
            action = self.store.decide_action(action_id, decision.decision, decision.action_digest)
        except ValueError:
            expired = self.store.get_action(action_id)
            if expired.status == ActionStatus.EXPIRED:
                self.store.update_task(expired.task_id, TaskStatus.FAILED, error="action approval expired", force=True)
            raise
        if decision.decision == "reject":
            self.store.update_task(action.task_id, TaskStatus.CANCELLED)
            return action
        self.store.update_task(action.task_id, TaskStatus.EXECUTING, phase=MapePhase.EXECUTE)
        self.store.update_action(action.id, ActionStatus.EXECUTING)
        self._spawn(self._run_execution_graph(action.task_id, action.id))
        return self.store.get_action(action.id)

    async def _reconcile_task(self, task_id: str) -> None:
        """Resume execution after a daemon restart."""
        task = await self._db(self.store.get_task, task_id)
        if task.status == TaskStatus.RECONCILING and task.phase == MapePhase.EXECUTE:
            actions = await self._db(self.store.list_actions, task_id)
            pending_action = next(
                (
                    action
                    for action in actions
                    if action.status in {ActionStatus.APPROVED, ActionStatus.EXECUTING}
                ),
                None,
            )
            if pending_action:
                self._spawn(self._run_execution_graph(task_id, pending_action.id))
            else:
                await self._db(
                    self.store.update_task,
                    task_id,
                    TaskStatus.FAILED,
                    error="no approved action to reconcile",
                    force=True,
                )
        else:
            await self._db(
                self.store.update_task,
                task_id,
                TaskStatus.FAILED,
                error="unrecoverable state after restart",
                force=True,
            )