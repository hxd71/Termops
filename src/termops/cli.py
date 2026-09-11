"""Termops CLI — terminal operations agent."""

from __future__ import annotations

import contextlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, cast

import click
import httpx
import tomli_w

from .config import Settings
from .llm import LLMProvider

TERMINAL_STATUSES = {"succeeded", "failed", "cancelled"}

SEVERITY_STYLES = {
    "critical": {"fg": "bright_red", "bold": True},
    "high": {"fg": "red"},
    "medium": {"fg": "yellow"},
    "low": {"fg": "green"},
}


class AgentClient:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.token = (
            settings.operator_token_path.read_text(encoding="utf-8").strip()
            if settings.operator_token_path.exists()
            else ""
        )
        if os.name != "nt" and settings.socket_path.exists():
            transport = httpx.HTTPTransport(uds=str(settings.socket_path))
            self.client = httpx.Client(transport=transport, base_url="http://termops.local", timeout=60)
        else:
            self.client = httpx.Client(base_url=f"http://{settings.web_host}:{settings.web_port}", timeout=60)

    def request(self, method: str, path: str, json_body: dict[str, Any] | None = None) -> Any:
        try:
            response = self.client.request(
                method, path, json=json_body, headers={"X-Operator-Token": self.token} if self.token else {}
            )
        except httpx.HTTPError as exc:
            raise click.ClickException(f"cannot reach the termops agent: {exc}") from exc
        if response.is_error:
            try:
                detail = response.json().get("detail", response.text)
            except ValueError:
                detail = response.text
            raise click.ClickException(f"agent returned HTTP {response.status_code}: {detail}")
        return response.json()


def _style_severity(sev: str) -> str:
    style: dict[str, Any] = cast(dict[str, Any], SEVERITY_STYLES.get(sev.lower(), {}))
    return click.style(f"[{sev.upper()}]", **style)


def _style_status(status: str) -> str:
    colors = {
        "succeeded": "green",
        "waiting_approval": "yellow",
        "failed": "red",
        "running": "cyan",
        "cancelled": "magenta",
    }
    fg = colors.get(status, "white")
    return click.style(status, fg=fg, bold=True)


def print_json(value: Any) -> None:
    click.echo(json.dumps(value, ensure_ascii=False, indent=2))


def _wait_task_detail(client: AgentClient, task_id: str, *, timeout: float = 180.0, interval: float = 0.5) -> dict[str, Any]:
    """Poll until the task settles, so the rendered report shows real results."""
    deadline = time.monotonic() + timeout
    while True:
        detail = client.request("GET", f"/v1/tasks/{task_id}")
        status = detail["task"]["status"]
        if status in TERMINAL_STATUSES or status == "waiting_approval" or time.monotonic() >= deadline:
            return detail
        time.sleep(interval)


def _render_task(detail: dict[str, Any]) -> None:
    """Human-readable rendering of a task detail."""
    task = detail["task"]
    report = task.get("report") or {}
    findings = detail.get("findings", [])
    actions = detail.get("actions", [])
    llm_attr = report.get("llm_attribution")

    click.echo(click.style("=" * 58, fg="bright_black"))
    click.echo(f"  Task:        {click.style(task['id'], fg='blue')}")
    click.echo(f"  Kind:        {task.get('kind', 'analyze')}")
    click.echo(f"  Status:      {_style_status(task['status'])}")
    click.echo(f"  Phase:       {task.get('phase', '-')}")
    click.echo(f"  Findings:    {len(findings)}")
    click.echo(f"  Actions:     {len(actions)}")
    click.echo(click.style("=" * 58, fg="bright_black"))

    if findings:
        click.echo(f"\n{click.style('Findings', bold=True, underline=True)}")
        for f in findings:
            sev = _style_severity(f.get("severity", "low"))
            click.echo(f"  {sev} {click.style(f.get('code', ''), bold=True)}: {f.get('meaning', '')}")
            if f.get("remediation"):
                click.echo(f"       {click.style(chr(8594), fg='green')} {f['remediation']}")

    if llm_attr:
        click.echo(f"\n{click.style('LLM Attribution', bold=True, underline=True)}")
        click.echo(f"  Cause:       {llm_attr.get('primary_cause', 'N/A')}")
        conf = llm_attr.get("confidence", 0)
        click.echo(f"  Confidence:  {conf:.0%}")
        for i, step in enumerate(llm_attr.get("remediation_steps", []), 1):
            click.echo(f"  Step {i}:     {step}")

    if actions:
        click.echo(f"\n{click.style('Proposed Actions', bold=True, underline=True)}")
        for a in actions:
            click.echo(f"  [{_style_status(a['status'])}] {a['id']}")
            payload = a.get("payload", {})
            if payload.get("command"):
                click.echo(f"       cmd:  {click.style(payload['command'], fg='cyan')}")
            click.echo(f"       hash: {a.get('digest', '-')[:16]}")

    if task["status"] == "waiting_approval":
        click.echo(f"\n{click.style('[!]', fg='yellow', bold=True)} Actions pending approval:")
        for a in actions:
            if a["status"] == "pending":
                click.echo(f"    termops action approve {a['id']}")

    if task["status"] == "failed":
        err = task.get("error", "")
        if err:
            click.echo(f"\n{click.style('Error', fg='red', bold=True)}: {err[:500]}")


@click.group()
@click.option("--config", "config_path", type=click.Path(path_type=Path), default=None)
@click.option("--profile", type=click.Choice(["live", "test", "demo"]), default=None)
@click.option("--json", "use_json", is_flag=True, default=False, help="Output raw JSON instead of formatted text.")
@click.pass_context
def cli(ctx: click.Context, config_path: Path | None, profile: str | None, use_json: bool) -> None:
    """Termops — terminal operations agent with LLM-powered error analysis."""
    settings = Settings.load(config_path, profile=profile)  # type: ignore[arg-type]
    ctx.obj = (AgentClient(settings), use_json, settings)


def _client_ctx(ctx: click.Context) -> tuple[AgentClient, bool]:
    return ctx.obj[0], ctx.obj[1]


def _settings_ctx(ctx: click.Context) -> Settings:
    return ctx.obj[2]


# ── Core analysis commands ────────────────────────────────────────────────


@cli.command()
@click.option("--text", default="", help="Inline error text to analyze.")
@click.option("--file", "file_path", type=click.Path(exists=True, dir_okay=False, path_type=Path), default=None)
@click.option("--source", default="stdin", help="Logical source label for the analyzed text.")
@click.option("--language", default="", help="Optional language or runtime label.")
@click.option("--command", default="", help="Optional command context.")
@click.option("--cwd", default="", help="Optional working directory context.")
@click.option("--exit-code", type=int, default=None, help="Optional exit code for the failing command.")
@click.pass_context
def analyze(
    ctx: click.Context,
    text: str,
    file_path: Path | None,
    source: str,
    language: str,
    command: str,
    cwd: str,
    exit_code: int | None,
) -> None:
    """Analyze terminal output or code failure text."""
    client, use_json = _client_ctx(ctx)
    if file_path is not None:
        text = file_path.read_text(encoding="utf-8")
        source = source or file_path.name
    elif not text:
        text = click.get_text_stream("stdin").read()
        source = source or "stdin"
    if not text.strip():
        raise click.ClickException("no analysis text provided")
    result = client.request(
        "POST",
        "/v1/tasks/analyze",
        {
            "text": text,
            "source": source,
            "language": language,
            "command": command,
            "cwd": cwd,
            "exit_code": exit_code,
        },
    )
    if use_json:
        print_json(result)
    else:
        _render_task(_wait_task_detail(client, result["id"]))


@cli.command(context_settings={"ignore_unknown_options": True})
@click.argument("command_parts", nargs=-1, type=str)
@click.option("--language", default="", help="Optional language or runtime label.")
@click.option("--cwd", default="", help="Optional working directory context.")
@click.pass_context
def run(ctx: click.Context, command_parts: tuple[str, ...], language: str, cwd: str) -> None:
    """Run a command locally and submit its output for analysis."""
    client, use_json = _client_ctx(ctx)
    if not command_parts:
        raise click.ClickException("run requires a command")
    result = client.request(
        "POST",
        "/v1/tasks/run",
        {"command": list(command_parts), "cwd": cwd, "language": language},
    )
    if use_json:
        print_json(result)
    else:
        _render_task(_wait_task_detail(client, result["id"]))


@cli.command()
@click.pass_context
def doctor(ctx: click.Context) -> None:
    """Show agent capability and environment checks."""
    client, use_json = _client_ctx(ctx)
    result = client.request("GET", "/v1/capabilities")
    if use_json:
        print_json(result)
    else:
        click.echo(click.style("Profile", bold=True) + f":  {result.get('profile', '-')}")
        llm = result.get("llm", {})
        provider = llm.get("provider", "none")
        llm_status = (
            click.style("enabled", fg="green") if llm.get("enabled") else click.style("disabled", fg="bright_black")
        )
        click.echo(click.style("LLM", bold=True) + f":     {llm_status} ({provider}/{llm.get('model') or 'none'})")
        host = result.get("host", {})
        click.echo(click.style("OS", bold=True) + f":      {host.get('os', '-')} / {host.get('arch', '-')}")
        click.echo(click.style("Shell", bold=True) + f":   {host.get('shell', '-')}")
        orch = result.get("orchestration", {})
        click.echo(click.style("MAPE-K", bold=True) + f":  {orch.get('framework', '-')}")


# ── Task management ────────────────────────────────────────────────────────


@cli.group()
def task() -> None:
    """Inspect agent tasks."""


@task.command("list")
@click.option("--limit", default=50, type=click.IntRange(1, 500))
@click.pass_context
def task_list(ctx: click.Context, limit: int) -> None:
    client, use_json = _client_ctx(ctx)
    result = client.request("GET", f"/v1/tasks?limit={limit}")
    if use_json:
        print_json(result)
    else:
        tasks = result if isinstance(result, list) else result.get("tasks", [])
        for t in tasks:
            sid = t["id"][:12]
            status = _style_status(t["status"])
            kind = t.get("kind", "analyze")
            click.echo(f"  {sid}  {status:30s}  {kind:10s}  {t.get('created_at', '')[:19]}")


@task.command("show")
@click.argument("task_id")
@click.pass_context
def task_show(ctx: click.Context, task_id: str) -> None:
    client, use_json = _client_ctx(ctx)
    result = client.request("GET", f"/v1/tasks/{task_id}")
    if use_json:
        print_json(result)
    else:
        _render_task(result)


@task.command("watch")
@click.argument("task_id")
@click.option("--interval", default=1.0, type=click.FloatRange(0.2, 10.0))
@click.pass_context
def task_watch(ctx: click.Context, task_id: str, interval: float) -> None:
    client, use_json = _client_ctx(ctx)
    if use_json:
        last_status = ""
        while True:
            detail = client.request("GET", f"/v1/tasks/{task_id}")
            status = detail["task"]["status"]
            if status != last_status:
                click.echo(f"{detail['task']['updated_at']}  {status}")
                last_status = status
            if status in TERMINAL_STATUSES or status == "waiting_approval":
                print_json(detail)
                return
            time.sleep(interval)
    else:
        last_status = ""
        while True:
            detail = client.request("GET", f"/v1/tasks/{task_id}")
            status = detail["task"]["status"]
            if status != last_status:
                click.echo(f"{detail['task']['updated_at']}  {_style_status(status)}")
                last_status = status
            if status in TERMINAL_STATUSES or status == "waiting_approval":
                _render_task(detail)
                return
            time.sleep(interval)


@task.command("cancel")
@click.argument("task_id")
@click.pass_context
def task_cancel(ctx: click.Context, task_id: str) -> None:
    """Cancel a queued/running/waiting task."""
    client, use_json = _client_ctx(ctx)
    result = client.request("POST", f"/v1/tasks/{task_id}/cancel")
    if use_json:
        print_json(result)
    else:
        click.echo(f"Task {task_id}: cancelled -> {_style_status(result.get('status', 'cancelled'))}")


# ── Action approval ────────────────────────────────────────────────────────


@cli.group()
def action() -> None:
    """Approve or reject immutable action proposals."""


def decide(ctx: click.Context, action_id: str, decision: str, yes: bool) -> None:
    client, use_json = _client_ctx(ctx)
    proposal = client.request("GET", f"/v1/actions/{action_id}")
    if use_json:
        print_json(proposal)
        if not yes and not click.confirm(f"{decision.title()} this exact action proposal?"):
            raise click.Abort()
    else:
        click.echo(f"Action: {proposal['kind']} -> {proposal['target']}")
        click.echo(f"Risk:   {click.style(proposal['risk'], fg='red' if proposal['risk'] == 'high' else 'yellow')}")
        click.echo(f"Digest: {proposal['digest']}")
        for step in proposal["steps"]:
            click.echo(f"  {step['order']}. {step['description']}")
        if not yes and not click.confirm(f"{decision.title()} this exact action proposal?"):
            raise click.Abort()
    result = client.request(
        "POST",
        f"/v1/actions/{action_id}/decision",
        {"decision": decision, "action_digest": proposal["digest"]},
    )
    if use_json:
        print_json(result)
    else:
        click.echo(f"Action {action_id}: {_style_status(result.get('status', decision))}")


@action.command("approve")
@click.argument("action_id")
@click.option("--yes", is_flag=True, help="Skip the local confirmation prompt.")
@click.pass_context
def action_approve(ctx: click.Context, action_id: str, yes: bool) -> None:
    decide(ctx, action_id, "approve", yes)


@action.command("reject")
@click.argument("action_id")
@click.option("--yes", is_flag=True, help="Skip the local confirmation prompt.")
@click.pass_context
def action_reject(ctx: click.Context, action_id: str, yes: bool) -> None:
    decide(ctx, action_id, "reject", yes)


# ── Web UI ──────────────────────────────────────────────────────────────────


@cli.group()
def web() -> None:
    """Manage the local Web UI session."""


@web.command("login")
@click.pass_context
def web_login(ctx: click.Context) -> None:
    client, _use_json = _client_ctx(ctx)
    result = client.request("POST", "/v1/web/login-code")
    click.echo(result["url"])
    click.echo("This one-time local link expires in 120 seconds.")


# ── Operator corrections ────────────────────────────────────────────────────


@cli.group()
def corrections() -> None:
    """Teach the classifier: manage operator corrections for error fingerprints."""


@corrections.command("add")
@click.argument("sample_text")
@click.option("--code", required=True, help="Classification code, e.g. POSTGRES_UNDEFINED_TABLE.")
@click.option(
    "--severity",
    type=click.Choice(["LOW", "MEDIUM", "HIGH", "CRITICAL"], case_sensitive=False),
    default="MEDIUM",
    show_default=True,
)
@click.option("--meaning", required=True, help="What this error means in your environment.")
@click.option("--remediation", default=None, help="How to fix it.")
@click.pass_context
def corrections_add(
    ctx: click.Context, sample_text: str, code: str, severity: str, meaning: str, remediation: str | None
) -> None:
    """Record a correction for the *kind* of error SAMPLE_TEXT represents.

    Paths, versions, IPs and line numbers are stripped into a normalized
    fingerprint, so one correction covers every recurrence of the same error.
    """
    client, use_json = _client_ctx(ctx)
    result = client.request(
        "POST",
        "/v1/corrections",
        {
            "sample_text": sample_text,
            "code": code,
            "severity": severity.upper(),
            "meaning": meaning,
            "remediation": remediation,
        },
    )
    if use_json:
        print_json(result)
    else:
        click.echo(f"Correction recorded for fingerprint {result['fingerprint']}")
        click.echo(f"  {_style_severity(result['severity'])} {result['code']}: {result['meaning']}")


@corrections.command("list")
@click.pass_context
def corrections_list(ctx: click.Context) -> None:
    """List recorded corrections, most-hit first."""
    client, use_json = _client_ctx(ctx)
    result = client.request("GET", "/v1/corrections")
    rows = sorted(result, key=lambda r: r.get("hits", 0), reverse=True)
    if use_json:
        print_json(rows)
        return
    if not rows:
        click.echo("No corrections recorded yet. Use `termops corrections add` to teach the classifier.")
        return
    for row in rows:
        click.echo(f"{_style_severity(row['severity'])} {row['code']}  (hits: {row['hits']})")
        click.echo(f"  fingerprint: {row['fingerprint']}")
        click.echo(f"  sample:      {row['sample'][:100]}")
        click.echo(f"  meaning:     {row['meaning']}")


@corrections.command("delete")
@click.argument("fingerprint")
@click.pass_context
def corrections_delete(ctx: click.Context, fingerprint: str) -> None:
    """Remove a correction by its fingerprint."""
    client, use_json = _client_ctx(ctx)
    result = client.request("DELETE", f"/v1/corrections/{fingerprint}")
    if use_json:
        print_json(result)
    else:
        click.echo(f"Correction {fingerprint}: deleted")


# ── Terminal hook ───────────────────────────────────────────────────────────


@cli.group()
def hook() -> None:
    """Manage the terminal auto-capture hook."""


@hook.command("install")
@click.pass_context
def hook_install(ctx: click.Context) -> None:
    """Install the terminal hook to auto-capture command errors."""
    settings = _settings_ctx(ctx)
    hooks_dir = Path(__file__).resolve().parent / "hooks"
    hook_file = settings.state_dir / "hook_status.txt"

    # Determine shell
    if os.name == "nt":
        shell = "powershell"
        script = hooks_dir / "hook.ps1"
        profile_path = Path(
            os.environ.get(
                "PROFILE",
                str(Path.home() / "Documents" / "WindowsPowerShell" / "Microsoft.PowerShell_profile.ps1"),
            )
        )
        source_line = f'. "{script.as_posix()}"'
    else:
        shell = settings.hook_shell if settings.hook_shell in {"bash", "zsh"} else os.environ.get("SHELL", "/bin/bash")
        script = hooks_dir / "hook.sh"
        profile_path = Path.home() / (".zshrc" if "zsh" in shell else ".bashrc")
        source_line = f'source "{script.as_posix()}"'

    # Write hook status
    hook_file.parent.mkdir(parents=True, exist_ok=True)
    hook_file.write_text("enabled")

    click.echo(f"Terminal hook installed for {shell}.")
    click.echo(f"  Hook script: {script}")
    click.echo(f"  Status file: {hook_file}")
    click.echo()
    click.echo(f"To activate, add this line to {profile_path}:")
    click.echo(f"  {source_line}")
    click.echo()
    click.echo("Or restart your terminal session.")


@hook.command("uninstall")
@click.pass_context
def hook_uninstall(ctx: click.Context) -> None:
    """Disable the terminal hook."""
    settings = _settings_ctx(ctx)
    hook_file = settings.state_dir / "hook_status.txt"
    hook_file.write_text("disabled")
    click.echo("Terminal hook disabled. Remove the source line from your shell profile to fully uninstall.")


@hook.command("status")
@click.pass_context
def hook_status(ctx: click.Context) -> None:
    """Check whether the terminal hook is active."""
    settings = _settings_ctx(ctx)
    hook_file = settings.state_dir / "hook_status.txt"
    if hook_file.exists():
        status = hook_file.read_text().strip()
        if status == "enabled":
            click.echo(click.style("Hook: enabled", fg="green"))
        else:
            click.echo(click.style("Hook: disabled", fg="yellow"))
    else:
        click.echo(click.style("Hook: not installed", fg="bright_black"))
        click.echo("Run 'termops hook install' to set up auto-capture.")


# ── LLM configuration ───────────────────────────────────────────────────────


@cli.group()
def config() -> None:
    """Configure LLM provider and agent settings."""


@config.command("show")
@click.pass_context
def config_show(ctx: click.Context) -> None:
    """Display current configuration."""
    settings = _settings_ctx(ctx)
    llm = settings.llm
    click.echo(click.style("Termops Configuration", bold=True))
    click.echo(f"  Config file:  {settings.config_file_path}")
    click.echo(f"  State dir:    {settings.state_dir}")
    click.echo(f"  Web:          {settings.web_host}:{settings.web_port}")
    click.echo()
    click.echo(click.style("LLM Provider", bold=True))
    click.echo(f"  Provider:     {llm.provider.value}")
    click.echo(f"  Enabled:      {'yes' if llm.enabled else 'no'}")
    click.echo(f"  Model:        {llm.model or '(not set)'}")
    click.echo(f"  Base URL:     {llm.base_url or '(default)'}")
    click.echo(f"  API Key:      {'[set]' if llm.api_key else '(not set)'}")
    click.echo(f"  Timeout:      {llm.timeout}s")
    click.echo(f"  Temperature:  {llm.temperature}")
    click.echo()

    click.echo(click.style("Terminal Hook", bold=True))
    click.echo(f"  Shell:        {settings.hook_shell}")
    hook_file = settings.state_dir / "hook_status.txt"
    hook_state = hook_file.read_text().strip() if hook_file.exists() else "not installed"
    click.echo(f"  Status:       {hook_state}")


@config.command("llm")
@click.option(
    "--provider",
    type=click.Choice([p.value for p in LLMProvider]),
    default=None,
    help="LLM provider (openai, anthropic, ollama, openai_compatible)",
)
@click.option("--model", default=None, help="Model name (e.g. gpt-4o, claude-sonnet-4-20250514, qwen2.5:7b)")
@click.option("--api-key", default=None, help="API key for the provider")
@click.option("--base-url", default=None, help="Override the default base URL")
@click.option("--enable/--disable", default=None, help="Enable or disable LLM analysis")
@click.pass_context
def config_llm(
    ctx: click.Context,
    provider: str | None,
    model: str | None,
    api_key: str | None,
    base_url: str | None,
    enable: bool | None,
) -> None:
    """Configure the LLM provider for error analysis."""
    settings = _settings_ctx(ctx)
    config_path = settings.config_file_path

    # Load existing config or create new
    if config_path.exists():
        with open(config_path, "rb") as f:
            if sys.version_info >= (3, 11):
                import tomllib
            else:
                import tomli as tomllib
            raw = tomllib.load(f)
    else:
        raw = {}

    llm_section = raw.get("llm", {})
    changed = False

    if provider is not None:
        llm_section["provider"] = provider
        changed = True
        click.echo(f"Provider set to: {provider}")

    if model is not None:
        llm_section["model"] = model
        changed = True
        click.echo(f"Model set to: {model}")

    if api_key is not None:
        llm_section["api_key"] = api_key
        changed = True
        click.echo("API key set (stored locally in config file)")

    if base_url is not None:
        llm_section["base_url"] = base_url
        changed = True
        click.echo(f"Base URL set to: {base_url}")

    if enable is not None:
        llm_section["enabled"] = enable
        changed = True
        click.echo(f"LLM analysis: {'enabled' if enable else 'disabled'}")

    if changed:
        raw["llm"] = llm_section
        config_path.parent.mkdir(parents=True, exist_ok=True)

        # Serialize the whole document (not just [llm]) so unrelated sections
        # such as [server], [operator], [policy], and [hook] survive a
        # `termops config llm` round-trip instead of being silently dropped.
        config_path.write_text(tomli_w.dumps(raw), encoding="utf-8")
        # The file may contain an API key: restrict to owner-only on POSIX.
        # On Windows chmod only toggles the read-only bit, which is a no-op here.
        with contextlib.suppress(OSError):
            config_path.chmod(0o600)
        click.echo(f"\nConfiguration saved to: {config_path}")
    else:
        click.echo("No changes specified. Use --help to see available options.")


# ── Evaluation ────────────────────────────────────────────────────────────────


@cli.group()
def eval() -> None:
    """Evaluate agent accuracy on benchmark datasets.

    Multi-tier evaluation following AgentBench (Liu et al., 2024, ICLR):
    Tier 1: Error code classification (Precision/Recall/F1)
    Tier 2: Root cause semantic similarity (BLEU, ROUGE-L)
    Tier 3: Action safety & validity
    Tier 4: End-to-end performance
    """


@eval.command("rule")
@click.option("--dataset", "-d", default="terminal_errors", help="Dataset name or path to JSONL.")
@click.pass_context
def eval_rule(ctx: click.Context, dataset: str) -> None:
    """Evaluate the deterministic rule engine on a dataset."""
    from .eval import EvalEngine, load_dataset, load_dataset_from_jsonl

    if Path(dataset).exists():
        samples = load_dataset_from_jsonl(dataset)
    else:
        try:
            samples = load_dataset(dataset)
        except ValueError as e:
            raise click.ClickException(str(e)) from e

    engine = EvalEngine()
    results = engine.evaluate_rule(samples)
    summary = engine.summarize(results)

    click.echo(f"Dataset: {dataset} ({len(samples)} samples)")
    click.echo(f"Code Accuracy: {summary.classification.accuracy:.1%}")
    click.echo(f"Micro F1: {summary.classification.micro_f1:.3f}")
    click.echo(f"Macro F1: {summary.classification.macro_f1:.3f}")
    click.echo(f"Avg BLEU-1: {summary.avg_bleu_1:.3f}")
    click.echo(f"Avg ROUGE-L: {summary.avg_rouge_l:.3f}")
    click.echo(f"Avg Latency: {summary.avg_latency_ms:.0f} ms")
    click.echo(f"Safety Violations: {summary.safety_violation_count}")

    # Per-category
    click.echo("\nPer-category:")
    for cat in sorted(summary.per_category):
        m = summary.per_category[cat]
        click.echo(f"  {cat}: {m['accuracy']:.1%} ({int(m['count'])} samples)")


@eval.command("run")
@click.option("--dataset", "-d", default="terminal_errors", help="Dataset name or path to JSONL.")
@click.option("--runs", "-n", default=1, type=int, help="Number of LLM runs for stochastic evaluation.")
@click.option("--output", "-o", default=None, help="Write Markdown report to file.")
@click.pass_context
def eval_run(ctx: click.Context, dataset: str, runs: int, output: str | None) -> None:
    """Run full evaluation including LLM attribution (if configured)."""
    from .eval import EvalEngine, load_dataset, load_dataset_from_jsonl

    if Path(dataset).exists():
        samples = load_dataset_from_jsonl(dataset)
    else:
        try:
            samples = load_dataset(dataset)
        except ValueError as e:
            raise click.ClickException(str(e)) from e

    engine = EvalEngine()

    # Rule engine evaluation (always deterministic)
    click.echo("=== Rule Engine Evaluation ===")
    rule_results = engine.evaluate_rule(samples)
    rule_summary = engine.summarize(rule_results)

    click.echo(f"Code Accuracy: {rule_summary.classification.accuracy:.1%}")
    click.echo(f"Micro F1: {rule_summary.classification.micro_f1:.3f}")
    click.echo(f"Macro F1: {rule_summary.classification.macro_f1:.3f}")
    click.echo(f"Avg BLEU-1: {rule_summary.avg_bleu_1:.3f}")
    click.echo(f"Avg ROUGE-L: {rule_summary.avg_rouge_l:.3f}")
    click.echo(f"Avg Latency: {rule_summary.avg_latency_ms:.0f} ms")
    click.echo(f"Safety Violations: {rule_summary.safety_violation_count}")

    # LLM evaluation (if configured)
    settings = _settings_ctx(ctx)
    if settings.llm.enabled:
        click.echo(f"\n=== LLM Evaluation ({runs} runs) ===")
        import asyncio

        from .llm_client import LLMClient

        llm = LLMClient(settings.llm)
        engine = EvalEngine(llm_client=llm)

        try:
            all_results = asyncio.get_event_loop().run_until_complete(engine.evaluate_llm(samples, runs=runs))
        except RuntimeError:
            all_results = asyncio.run(engine.evaluate_llm(samples, runs=runs))

        llm_summary = engine.summarize_llm(all_results)
        click.echo(f"Code Accuracy (mean): {llm_summary['code_accuracy_mean']:.1%}")
        click.echo(f"Code Accuracy (std):  {llm_summary['code_accuracy_std']:.3f}")
        click.echo(f"ROUGE-L (mean): {llm_summary['rouge_l_mean']:.3f}")
        click.echo(f"ROUGE-L (std):  {llm_summary['rouge_l_std']:.3f}")
        click.echo(f"Latency (mean): {llm_summary['latency_ms_mean']:.0f} ms")
        click.echo(f"Latency (std):  {llm_summary['latency_ms_std']:.0f} ms")
    else:
        click.echo("\nLLM not configured. Set TERMOPS_LLM_ENABLED=true to enable LLM evaluation.")

    # Write report
    if output:
        report = rule_summary.to_markdown()
        Path(output).write_text(report, encoding="utf-8")
        click.echo(f"\nReport saved to: {output}")


@eval.command("list")
def eval_list() -> None:
    """List available built-in evaluation datasets."""
    from .eval import list_datasets

    for name in list_datasets():
        click.echo(f"  {name}")


def main() -> None:
    cli()


if __name__ == "__main__":
    main()