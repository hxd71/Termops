"""LangGraph orchestration layer for the MAPE-K error analysis loop.

The graph splits reasoning (monitor -> classify -> plan) from execution
(execute -> observe -> knowledge) and surfaces the approval wait as an
explicit terminal edge. All side effects remain auditable through the
persistent `StateStore` owned by the engine.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, TypedDict

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from .diagnostics import classify_command_risk, suggest_followup_command
from .models import (
    ActionStatus,
    ActionStep,
    AnalysisRequest,
    LLMAttribution,
    MapePhase,
    RiskLevel,
    Severity,
    TaskStatus,
)


class OpsGraphState(TypedDict, total=False):
    """Mutable state passed between MAPE-K graph nodes."""

    task_id: str
    request: dict[str, Any]
    env: dict[str, Any] | None
    classification: dict[str, Any]
    retrieved_knowledge: list[dict[str, Any]]
    llm_attribution: dict[str, Any] | None
    proposed_command: str | None
    action_id: str | None
    command_result: dict[str, Any] | None
    summary: str | None
    terminal_status: str | None
    error: str | None


def _engine(config: RunnableConfig) -> Any:
    """Pull the engine instance out of the LangGraph config."""
    return config["configurable"]["engine"]


async def _monitor_node(state: OpsGraphState, config: RunnableConfig) -> dict[str, Any]:
    engine = _engine(config)
    task_id = state["task_id"]
    await engine._db(engine._mark_phase, task_id, MapePhase.MONITOR, "capturing input context")
    request, env = await engine._monitor(task_id)
    return {"request": request.model_dump(mode="json"), "env": env.model_dump(mode="json")}


async def _analyze_node(state: OpsGraphState, config: RunnableConfig) -> dict[str, Any]:
    engine = _engine(config)
    task_id = state["task_id"]
    await engine._db(engine._mark_phase, task_id, MapePhase.ANALYZE, "pattern matching and finding extraction")

    request = AnalysisRequest.model_validate(state["request"])
    # Operator corrections (feedback loop) take precedence over built-in rules.
    classification = await engine._db(engine.classify, request.text, request.exit_code)

    observations = await engine._db(engine.store.list_observations, task_id)
    evidence_ids = [observations[0].id] if observations else []
    for match in classification["findings"]:
        await engine._db(
            engine.store.add_finding,
            task_id,
            match["code"],
            Severity(str(match["severity"]).lower()),
            0.9,
            match["meaning"],
            f"Line {match['line']}: {match['text']}",
            evidence_ids,
            match["remediation"],
        )
    return {"classification": classification}


async def _retrieve_node(state: OpsGraphState, config: RunnableConfig) -> dict[str, Any]:
    """Search the knowledge base for similar past error patterns."""
    engine = _engine(config)
    task_id = state["task_id"]
    await engine._db(engine._mark_phase, task_id, MapePhase.ANALYZE, "retrieving relevant knowledge")

    request = AnalysisRequest.model_validate(state["request"])
    codes = state["classification"].get("codes", [])
    # Build a search query from the error text and detected codes.
    query_parts = [request.text[:300]]
    if codes:
        query_parts.append(" ".join(codes))
    query = " ".join(query_parts)

    chunks = await engine._db(engine.store.search_knowledge, query, 5)
    if not chunks and request.text:
        # Fallback: search by first meaningful line of error text.
        first_line = request.text.strip().split("\n")[0][:200]
        if first_line:
            chunks = await engine._db(engine.store.search_knowledge, first_line, 3)

    # Operator-pinned history tasks take precedence over FTS similarity hits.
    history_chunks: list[dict[str, Any]] = []
    for hid in request.history_task_ids[:5]:
        try:
            history_chunks.extend(await engine._db(engine.store.list_knowledge, hid))
        except KeyError:
            continue
    seen: set[str] = set()
    merged: list[dict[str, Any]] = []
    for chunk in [*history_chunks, *chunks]:
        cid = chunk.get("id")
        if cid and cid not in seen:
            seen.add(cid)
            merged.append(chunk)
    return {"retrieved_knowledge": merged[:5]}


async def _llm_attribution_node(state: OpsGraphState, config: RunnableConfig) -> dict[str, Any]:
    """Ask the LLM for structured error attribution when available."""
    engine = _engine(config)
    task_id = state["task_id"]
    llm = getattr(engine, "llm", None)
    if not llm or not llm.enabled:
        return {"llm_attribution": None}

    await engine._db(engine._mark_phase, task_id, MapePhase.ANALYZE, "llm attribution in progress")
    request = AnalysisRequest.model_validate(state["request"])
    attribution = await llm.attribute_error(
        text=request.text,
        command=request.command,
        language=request.language,
        exit_code=request.exit_code,
        env=state.get("env") or {},
        findings=state["classification"].get("findings", []),
        retrieved_chunks=state.get("retrieved_knowledge", []),
    )
    if attribution:
        await engine._db(
            engine.store.append_event,
            "llm.attribution",
            {
                "primary_cause": attribution.primary_cause,
                "confidence": attribution.confidence,
                "remediation_steps": attribution.remediation_steps,
                "proposed_command": attribution.proposed_command,
                "needs_approval": attribution.needs_approval,
                "safety_notes": attribution.safety_notes,
            },
            task_id,
        )
    return {"llm_attribution": attribution.model_dump(mode="json") if attribution else None}


async def _plan_node(state: OpsGraphState, config: RunnableConfig) -> dict[str, Any]:
    engine = _engine(config)
    task_id = state["task_id"]
    await engine._db(engine._mark_phase, task_id, MapePhase.PLAN, "building remediation proposals")

    request = AnalysisRequest.model_validate(state["request"])
    codes = set(state["classification"].get("codes", []))

    # Prefer LLM attribution's proposed command when available.
    llm_attr = state.get("llm_attribution")
    rule_suggested = suggest_followup_command(request.text, codes, request.language, request.command)
    suggested = (llm_attr.get("proposed_command") if llm_attr else None) or rule_suggested

    if not suggested:
        return {"proposed_command": None}

    risk = {"low": RiskLevel.LOW, "medium": RiskLevel.MEDIUM, "high": RiskLevel.HIGH}[
        classify_command_risk(suggested)
    ]
    action = await engine._db(
        engine.store.create_action,
        task_id,
        "run_command",
        suggested,
        risk,
        {
            "command": suggested,
            "requested_command": request.command,
            "cwd": request.cwd,
            "language": request.language,
            "source": request.source,
            "safety_notes": (llm_attr.get("safety_notes") if llm_attr else []) or [],
        },
        [
            ActionStep(order=1, action="run_command", description="Run the proposed command locally"),
            ActionStep(order=2, action="capture_output", description="Capture stdout, stderr, and exit code"),
            ActionStep(order=3, action="analyze_output", description="Record the result for review"),
        ],
        ["The command proposal is scoped to debugging context only"],
        ["A command result is returned and stored"],
        ["No state mutation is applied beyond local command execution"],
        engine.settings.approval_ttl_seconds,
    )
    return {"proposed_command": suggested, "action_id": action.id}


def _route_plan(state: OpsGraphState) -> str:
    return "wait_approval" if state.get("proposed_command") else "finish"


async def _wait_approval_node(state: OpsGraphState, config: RunnableConfig) -> dict[str, Any]:
    engine = _engine(config)
    task_id = state["task_id"]
    await engine._db(engine._mark_phase, task_id, MapePhase.PLAN, "waiting for operator approval")
    report = await engine._db(engine._build_report, task_id, "A command proposal is ready for approval.")
    await engine._db(
        engine.store.update_task,
        task_id,
        TaskStatus.WAITING_APPROVAL,
        phase=MapePhase.PLAN,
        report=report,
    )
    return {"terminal_status": "waiting_approval"}


async def _finish_node(state: OpsGraphState, config: RunnableConfig) -> dict[str, Any]:
    engine = _engine(config)
    task_id = state["task_id"]
    request = AnalysisRequest.model_validate(state["request"])
    llm_attr = state.get("llm_attribution")
    await engine._finish_analysis(
        task_id,
        state.get("classification", {}),
        request,
        llm_attribution=LLMAttribution.model_validate(llm_attr) if llm_attr else None,
        retrieved_knowledge_ids=[chunk["id"] for chunk in (state.get("retrieved_knowledge") or [])],
    )
    return {"terminal_status": "succeeded"}


async def _execute_node(state: OpsGraphState, config: RunnableConfig) -> dict[str, Any]:
    engine = _engine(config)
    task_id = state["task_id"]
    action_id = state["action_id"]
    await engine._db(engine._mark_phase, task_id, MapePhase.EXECUTE, "running approved command")

    action = await engine._db(engine.store.get_action, action_id)
    command = str(action.payload.get("command", "")).strip()
    cwd = str(action.payload.get("cwd", "")).strip() or None
    if cwd and not Path(cwd).exists():
        cwd = None
    if not command:
        raise ValueError("run_command action is missing a command")

    result = await engine._run_command(command, cwd)
    return {"command_result": result}


async def _observe_node(state: OpsGraphState, config: RunnableConfig) -> dict[str, Any]:
    engine = _engine(config)
    task_id = state["task_id"]
    action_id = state["action_id"]
    result = state.get("command_result")
    assert result is not None, "command_result must be set before observe node"
    await engine._db(engine._mark_phase, task_id, MapePhase.OBSERVE, "capturing command output")

    observation = await engine._db(
        engine.store.add_observation,
        task_id,
        "command_result",
        "approved command execution",
        "ok" if result["returncode"] == 0 else "warning",
        f"Approved command exited with code {result['returncode']}.",
        result,
    )
    result_text = result["stdout"] + "\n" + result["stderr"]
    result_classification = await engine._db(engine.classify, result_text, result["returncode"])
    for match in result_classification["findings"]:
        await engine._db(
            engine.store.add_finding,
            task_id,
            match["code"],
            Severity(str(match["severity"]).lower()),
            0.85,
            match["meaning"],
            f"Line {match['line']}: {match['text']}",
            [observation.id],
            match["remediation"],
        )
    await engine._db(
        engine.store.update_action,
        action_id,
        ActionStatus.SUCCEEDED,
        {"ok": result["returncode"] == 0, **result},
    )
    return {}


async def _knowledge_node(state: OpsGraphState, config: RunnableConfig) -> dict[str, Any]:
    engine = _engine(config)
    task_id = state["task_id"]
    result = state.get("command_result")
    assert result is not None, "command_result must be set before knowledge node"

    await engine._db(engine._mark_phase, task_id, MapePhase.OBSERVE, "verifying command result")
    await engine._db(engine.store.update_task, task_id, TaskStatus.VERIFYING, phase=MapePhase.OBSERVE)

    await engine._db(engine._mark_phase, task_id, MapePhase.KNOWLEDGE, "recording command execution outcome")
    summary = f"Approved command completed with exit code {result['returncode']}."
    await engine._db(engine.store.add_knowledge, task_id, "command_result", summary, result)
    report = await engine._db(engine._build_report, task_id, summary)
    await engine._db(
        engine.store.update_task,
        task_id,
        TaskStatus.SUCCEEDED,
        phase=MapePhase.KNOWLEDGE,
        report=report,
    )
    return {"terminal_status": "succeeded"}


def build_analysis_graph() -> CompiledStateGraph:
    """Build the reasoning graph: monitor -> classify -> retrieve -> llm -> plan -> wait/finish."""
    builder = StateGraph(OpsGraphState)
    builder.add_node("monitor", _monitor_node)
    builder.add_node("analyze", _analyze_node)
    builder.add_node("retrieve", _retrieve_node)
    builder.add_node("llm_attribution", _llm_attribution_node)
    builder.add_node("plan", _plan_node)
    builder.add_node("wait_approval", _wait_approval_node)
    builder.add_node("finish", _finish_node)
    builder.add_edge(START, "monitor")
    builder.add_edge("monitor", "analyze")
    builder.add_edge("analyze", "retrieve")
    builder.add_edge("retrieve", "llm_attribution")
    builder.add_edge("llm_attribution", "plan")
    builder.add_conditional_edges("plan", _route_plan, {"wait_approval": "wait_approval", "finish": "finish"})
    builder.add_edge("wait_approval", END)
    builder.add_edge("finish", END)
    return builder.compile()


def build_execution_graph() -> CompiledStateGraph:
    """Build the execution graph: execute -> observe -> knowledge."""
    builder = StateGraph(OpsGraphState)
    builder.add_node("execute", _execute_node)
    builder.add_node("observe", _observe_node)
    builder.add_node("knowledge", _knowledge_node)
    builder.add_edge(START, "execute")
    builder.add_edge("execute", "observe")
    builder.add_edge("observe", "knowledge")
    builder.add_edge("knowledge", END)
    return builder.compile()