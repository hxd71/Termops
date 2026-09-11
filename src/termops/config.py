"""Termops settings loaded from config file, env vars, and CLI overrides."""

from __future__ import annotations

import ipaddress
import os
import sys

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Literal, cast

from .llm import PROVIDER_DEFAULTS, LLMConfig, LLMProvider

Profile = Literal["live", "test", "demo"]


@dataclass(frozen=True)
class Settings:
    """Termops agent configuration.

    Loaded from (in priority order):
    1. CLI flags (--config, --profile)
    2. Environment variables (TERMOPS_*)
    3. Config file (~/.termops/config.toml)
    4. Built-in defaults
    """

    profile: Profile = "live"
    state_dir: Path = Path("~/.termops").expanduser()
    config_dir: Path = Path("~/.termops/config").expanduser()
    run_dir: Path = Path("~/.termops/run").expanduser()
    web_host: str = "127.0.0.1"
    web_port: int = 8923
    web_secure_cookie: bool = False  # enable when serving the UI behind HTTPS
    approval_ttl_seconds: int = 900
    session_ttl_seconds: int = 8 * 60 * 60
    operator_name: str = "local-operator"

    # ── LLM provider settings ──
    llm: LLMConfig = field(default_factory=LLMConfig)

    # ── Terminal hook settings ──
    hook_shell: str = "auto"  # auto, powershell, bash, zsh

    env_allowlist: set[str] = field(
        default_factory=lambda: {
            "PATH",
            "HOME",
            "USER",
            "SHELL",
            "OS",
            "PYTHONPATH",
            "VIRTUAL_ENV",
            "CONDA_DEFAULT_ENV",
            "NODE_PATH",
            "GO111MODULE",
            "RUST_BACKTRACE",
            "LANG",
        }
    )
    extra: dict[str, Any] = field(default_factory=dict)

    # ── Path properties ──
    @property
    def database_path(self) -> Path:
        return self.state_dir / "state.db"

    @property
    def backup_dir(self) -> Path:
        return self.state_dir / "backups"

    @property
    def socket_path(self) -> Path:
        return self.run_dir / "agent.sock"

    @property
    def operator_token_path(self) -> Path:
        return self.state_dir / "operator.token"

    @property
    def config_file_path(self) -> Path:
        return self.config_dir / "config.toml"

    def validate(self) -> Settings:
        try:
            host = ipaddress.ip_address(self.web_host)
        except ValueError as exc:
            raise ValueError("web_host must be a valid IP address") from exc
        if not host.is_loopback:
            raise ValueError("Web UI refuses non-loopback bindings")
        if not 1 <= self.web_port <= 65535:
            raise ValueError("web_port must be between 1 and 65535")
        if self.profile not in {"live", "test", "demo"}:
            raise ValueError(f"unsupported profile: {self.profile}")
        return self

    @classmethod
    def load(cls, config_path: str | Path | None = None, profile: Profile | None = None) -> Settings:
        selected_profile = profile or os.environ.get("TERMOPS_PROFILE", os.environ.get("ERRA_PROFILE", "live"))
        if selected_profile not in {"live", "test", "demo"}:
            raise ValueError(f"unsupported profile: {selected_profile}")

        home = Path(os.environ.get("TERMOPS_HOME", os.environ.get("ERRA_HOME", Path.home() / ".termops"))).resolve()
        settings = cls(
            profile=cast(Profile, selected_profile),
            state_dir=home,
            config_dir=home / "config",
            run_dir=home / "run",
        )

        def _env(key: str, default: str = "") -> str:
            return os.environ.get(key, default)

        def _env_bool(key: str, default: bool = False) -> bool:
            value = os.environ.get(key, "")
            if value.lower() in {"1", "true", "yes", "on"}:
                return True
            if value.lower() in {"0", "false", "no", "off"}:
                return False
            return default

        # ── Config file ──
        path = Path(config_path).resolve() if config_path else settings.config_file_path

        if path.exists():
            with path.open("rb") as cf:
                raw = tomllib.load(cf)

            server = raw.get("server", {})
            policy = raw.get("policy", {})
            llm_raw = raw.get("llm", {})
            hook_raw = raw.get("hook", {})

            # LLM provider
            provider_name = _env("TERMOPS_LLM_PROVIDER", str(llm_raw.get("provider", "ollama")))
            try:
                provider = LLMProvider(provider_name)
            except ValueError:
                provider = LLMProvider.OLLAMA

            defaults = PROVIDER_DEFAULTS.get(provider, {})
            llm_config = LLMConfig(
                provider=provider,
                enabled=_env_bool("TERMOPS_LLM_ENABLED", bool(llm_raw.get("enabled", False))),
                api_key=_env("TERMOPS_LLM_API_KEY", str(llm_raw.get("api_key", ""))),
                base_url=_env("TERMOPS_LLM_BASE_URL", str(llm_raw.get("base_url", defaults.get("base_url", "")))),
                model=_env("TERMOPS_LLM_MODEL", str(llm_raw.get("model", defaults.get("model", "")))),
                timeout=float(_env("TERMOPS_LLM_TIMEOUT", str(llm_raw.get("timeout", "60")))),
                temperature=float(_env("TERMOPS_LLM_TEMPERATURE", str(llm_raw.get("temperature", "0.2")))),
                max_tokens=int(_env("TERMOPS_LLM_MAX_TOKENS", str(llm_raw.get("max_tokens", "4096")))),
                extra_headers=dict(llm_raw.get("extra_headers", {})),
                extra=llm_raw.get("extra", {}),
            )

            settings = replace(
                settings,
                web_host=str(server.get("web_host", settings.web_host)),
                web_port=int(server.get("web_port", settings.web_port)),
                web_secure_cookie=bool(server.get("web_secure_cookie", settings.web_secure_cookie)),
                state_dir=Path(server.get("state_dir", str(settings.state_dir))).expanduser().resolve(),
                run_dir=Path(server.get("run_dir", str(settings.run_dir))).expanduser().resolve(),
                approval_ttl_seconds=int(policy.get("approval_ttl_seconds", settings.approval_ttl_seconds)),
                env_allowlist=set(policy.get("env_allowlist", list(settings.env_allowlist))),
                operator_name=str(raw.get("operator", {}).get("name", settings.operator_name)),
                llm=llm_config,
                hook_shell=str(hook_raw.get("shell", settings.hook_shell)),
                extra=raw,
            )
        else:
            # No config file — use env vars only
            provider_name = _env("TERMOPS_LLM_PROVIDER", "ollama")
            try:
                provider = LLMProvider(provider_name)
            except ValueError:
                provider = LLMProvider.OLLAMA

            defaults = PROVIDER_DEFAULTS.get(provider, {})
            llm_config = LLMConfig(
                provider=provider,
                enabled=_env_bool("TERMOPS_LLM_ENABLED", False),
                api_key=_env("TERMOPS_LLM_API_KEY", ""),
                base_url=_env("TERMOPS_LLM_BASE_URL", defaults.get("base_url", "")),
                model=_env("TERMOPS_LLM_MODEL", defaults.get("model", "")),
                timeout=float(_env("TERMOPS_LLM_TIMEOUT", "60")),
                temperature=float(_env("TERMOPS_LLM_TEMPERATURE", "0.2")),
                max_tokens=int(_env("TERMOPS_LLM_MAX_TOKENS", "4096")),
            )
            settings = replace(settings, llm=llm_config)

        return settings.validate()

    def ensure_directories(self) -> None:
        for path in (self.state_dir, self.run_dir, self.backup_dir, self.config_dir):
            path.mkdir(parents=True, exist_ok=True)
        if os.name != "nt":
            self.state_dir.chmod(0o711)
            self.run_dir.chmod(0o711)
            self.backup_dir.chmod(0o750)