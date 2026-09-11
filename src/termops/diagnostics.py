"""Generic terminal/code error classification.

This module is intentionally deterministic: it uses structured pattern matching to
produce findings that can be tested, versioned, and audited. LLM-based enrichment
is left as an optional plugin layer outside the production core.
"""

from __future__ import annotations

import os
import re
import sys
from typing import Any

from .models import Severity

# Normalizes volatile fragments (paths, versions, hex ids, line numbers) out of
# error text so that operator corrections can be keyed by a stable fingerprint.
_NORMALIZE_PATTERNS = (
    (re.compile(r"[A-Za-z]:\\[^\s:'\"]+"), "<PATH>"),  # Windows paths
    (re.compile(r"(?<![\w/])/(?:[\w.\-]+/)+[\w.\-]+"), "<PATH>"),  # POSIX paths
    (re.compile(r"\b\d+\.\d+(?:\.\d+)*(?:[-+][\w.]+)?\b"), "<VER>"),
    (re.compile(r"\b[0-9a-f]{7,40}\b", re.IGNORECASE), "<HEX>"),
    (re.compile(r"\bline\s+\d+\b", re.IGNORECASE), "line <N>"),
    (re.compile(r":\d+(:\d+)?"), ":<N>"),
    (re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}\b"), "<IP>"),
    (re.compile(r"\s+"), " "),
)


def normalize_error_text(text: str) -> str:
    """Return a stable, lowercase fingerprint of an error message.

    Volatile details (paths, version numbers, hashes, line numbers, IPs) are
    replaced with placeholders so that two occurrences of the *same kind* of
    error map to one key. Used by the operator-correction feedback loop.
    """
    out = text.lower()
    for pattern, repl in _NORMALIZE_PATTERNS:
        out = pattern.sub(repl, out)
    return out.strip()[:500]


ErrorPattern = tuple[tuple[str, ...], Severity, str, str]

GENERIC_ERROR_PATTERNS: dict[str, ErrorPattern] = {
    "COMMAND_NOT_FOUND": (
        (
            "command not found",
            "is not recognized as an internal or external command",
            "is not recognized",
            "was not found in this repository",
        ),
        Severity.HIGH,
        "The shell could not resolve the command name.",
        "Check PATH, shell spelling, and platform-specific command syntax.",
    ),
    "MODULE_NOT_FOUND": (
        (
            "module not found",
            "cannot find module",
            "no module named",
            "modulenotfounderror",
            "cannot import name",
        ),
        Severity.HIGH,
        "The runtime could not import a required dependency.",
        "Install the missing package or verify the active interpreter/environment.",
    ),
    "PACKAGE_NOT_FOUND": (
        (
            "package was not found",
            "cannot find package",
            "error: cannot find module",
            "cannot resolve package",
            "npm err! code enoent",
            "could not find package",
        ),
        Severity.HIGH,
        "A package manager could not locate the declared package.",
        "Verify the package name, registry, lockfile, and network connectivity.",
    ),
    "PERMISSION_DENIED": (
        (
            "permission denied",
            "access is denied",
            "operation not permitted",
            "errno 13",
            "unauthorized",
        ),
        Severity.HIGH,
        "The process hit a filesystem or OS permission boundary.",
        "Inspect file ownership, ACLs, sudo requirements, and execution context.",
    ),
    "FILE_NOT_FOUND": (
        (
            "no such file or directory",
            "file not found",
            "cannot open file",
            "enoent",
            "path does not exist",
        ),
        Severity.MEDIUM,
        "A referenced file or path was missing.",
        "Verify relative paths, working directory, and generated artifact locations.",
    ),
    "SYNTAX_ERROR": (
        (
            "syntaxerror",
            "unexpected token",
            "invalid syntax",
            "syntax error",
            "unexpected indent",
        ),
        Severity.HIGH,
        "The source file or command text contains a syntax problem.",
        "Check the parser line and reduce the input to the smallest failing example.",
    ),
    "TEST_FAILURE": (
        (
            "assertionerror",
            "assertion failed",
            "assert expected",
            "tests failed",
            "test failure",
            "failed tests",
        ),
        Severity.MEDIUM,
        "A test assertion or verification step failed.",
        "Inspect the assertion diff, fixture setup, and recent code changes.",
    ),
    "TYPE_ERROR": (
        (
            "typeerror",
            "type error",
            "cannot be applied to operands of type",
            "incompatible types",
            "mismatched types",
        ),
        Severity.MEDIUM,
        "A runtime or compile-time type mismatch was detected.",
        "Check argument types, generics, and recent API signature changes.",
    ),
    "NETWORK_FAILURE": (
        (
            "connection refused",
            "connection timed out",
            "could not resolve host",
            "network is unreachable",
            "econnrefused",
            "err_name_not_resolved",
            "temporary failure in name resolution",
        ),
        Severity.HIGH,
        "A network-dependent operation could not reach its target.",
        "Verify DNS, proxy, firewall, VPN, and target service availability.",
    ),
    "PORT_IN_USE": (
        (
            "address already in use",
            "port is already in use",
            "eaddrinuse",
            "bind: address already in use",
        ),
        Severity.MEDIUM,
        "A local socket bind failed because the port is occupied.",
        "Identify the conflicting process or configure a different port.",
    ),
    "ENV_VAR_MISSING": (
        (
            "environment variable",
            "env var",
            "keyerror: '",
            "required environment variable",
            "is not set",
        ),
        Severity.MEDIUM,
        "An expected environment variable is missing or empty.",
        "Set the variable in the current shell or .env file and retry.",
    ),
    "DEPENDENCY_CONFLICT": (
        (
            "version conflict",
            "incompatible with",
            "dependency resolution failed",
            "conflicting dependencies",
            "peer dep",
        ),
        Severity.HIGH,
        "Installed dependencies have incompatible version requirements.",
        "Review lockfiles, constraints, and consider a clean install.",
    ),
    "BUILD_FAILURE": (
        (
            "build failed",
            "compilation failed",
            "failed to compile",
            "error: command 'gcc'",
            "exit status 1",
            "ld: symbol(s) not found",
        ),
        Severity.HIGH,
        "A build or compilation step failed.",
        "Inspect compiler output, missing headers/libs, and build environment.",
    ),
    "STACK_OVERFLOW": (
        (
            "maximum call stack size exceeded",
            "recursionerror",
            "stack overflow",
        ),
        Severity.HIGH,
        "Infinite recursion or excessive call depth was detected.",
        "Check termination conditions and recursive function logic.",
    ),
    "MEMORY_EXHAUSTED": (
        (
            "out of memory",
            "memoryerror",
            "killed process",
            "cannot allocate memory",
            "oom-kill",
            "oomkilled",
            "oom killer",
        ),
        Severity.CRITICAL,
        "The process exhausted available memory.",
        "Reduce data size, batch size, or memory reservations before retrying.",
    ),
    "EXIT_NON_ZERO": (
        (
            "exited with code",
            "exit status",
            "returned non-zero",
            "process finished with exit code",
        ),
        Severity.MEDIUM,
        "A command or process exited with a non-zero status.",
        "Inspect the preceding stderr/stdout for the underlying cause.",
    ),
    # ── Docker ────────────────────────────────────────────────────────
    "DOCKER_NOT_RUNNING": (
        (
            "docker daemon is not running",
            "cannot connect to the docker daemon",
            "is the docker daemon running",
            "error during connect",
        ),
        Severity.HIGH,
        "The Docker daemon is not reachable.",
        "Start Docker Desktop or the dockerd service, and verify the DOCKER_HOST variable.",
    ),
    "DOCKER_IMAGE_NOT_FOUND": (
        (
            "pull access denied",
            "repository does not exist",
            "manifest unknown",
            "not found: manifest",
            "image not found",
        ),
        Severity.HIGH,
        "A Docker image or repository could not be located.",
        "Verify the image tag, registry login, and network connectivity.",
    ),
    "DOCKER_BUILD_FAILED": (
        (
            "docker build failed",
            "the command '/bin/sh -c' returned a non-zero code",
            "error building image",
            "failed to build",
        ),
        Severity.HIGH,
        "A Docker build step failed.",
        "Inspect the failing RUN/COPY layer, check Dockerfile syntax and base image availability.",
    ),
    # ── Git ───────────────────────────────────────────────────────────
    "GIT_AUTH_FAILED": (
        (
            "authentication failed",
            "fatal: authentication failed",
            "remote: invalid username or password",
            "could not read from remote repository",
            "permission denied (publickey)",
            "please make sure you have the correct access rights",
        ),
        Severity.HIGH,
        "Git authentication to the remote repository failed.",
        "Verify SSH key, personal access token, or credential helper configuration.",
    ),
    "GIT_MERGE_CONFLICT": (
        (
            "merge conflict",
            "automatic merge failed",
            "conflict: merge",
            "both modified:",
            "unmerged paths",
        ),
        Severity.MEDIUM,
        "A git merge or rebase encountered conflicts.",
        "Resolve conflicts in the listed files, then git add and commit.",
    ),
    "GIT_REMOTE_REJECTED": (
        (
            "remote rejected",
            "failed to push some refs",
            "non-fast-forward",
            "updates were rejected",
        ),
        Severity.MEDIUM,
        "A git push was rejected by the remote.",
        "Pull latest changes first, resolve conflicts, or check branch protection rules.",
    ),
    # ── SSL / TLS ─────────────────────────────────────────────────────
    "SSL_CERT_ERROR": (
        (
            "certificate verify failed",
            "ssl certificate",
            "self-signed certificate",
            "unable to get local issuer certificate",
            "certificate has expired",
            "ssl: certificate",
        ),
        Severity.HIGH,
        "An SSL/TLS certificate verification failed.",
        "Check the certificate chain, expiration date, or add the CA to the trust store.",
    ),
    # ── Timeout ───────────────────────────────────────────────────────
    "TIMEOUT_ERROR": (
        (
            "timed out",
            "timeout",
            "deadline exceeded",
            "context deadline exceeded",
            "request timed out",
        ),
        Severity.MEDIUM,
        "An operation exceeded its time limit.",
        "Increase the timeout setting, check network latency, or optimize the operation.",
    ),
    # ── Disk ──────────────────────────────────────────────────────────
    "DISK_FULL": (
        (
            "no space left on device",
            "disk full",
            "enospc",
            "insufficient disk space",
            "quota exceeded",
        ),
        Severity.CRITICAL,
        "The filesystem has run out of available space.",
        "Free disk space, prune logs/caches, or extend the volume.",
    ),
    # ── Rate Limit ────────────────────────────────────────────────────
    "RATE_LIMITED": (
        (
            "rate limit exceeded",
            "too many requests",
            "429",
            "rate limited",
            "api rate limit",
            "throttled",
        ),
        Severity.MEDIUM,
        "An API or service call was rate-limited.",
        "Wait for the rate window to reset, or check quota/plan limits.",
    ),
    # ── Config / YAML ─────────────────────────────────────────────────
    "CONFIG_PARSE_ERROR": (
        (
            "yaml.scanner",
            "yaml.parser",
            "while parsing",
            "could not find expected",
            "mapping values are not allowed",
            "duplicate key",
            "invalid yaml",
        ),
        Severity.MEDIUM,
        "A configuration file failed to parse (likely YAML indentation or syntax).",
        "Validate the YAML/JSON with a linter and fix indentation or duplicate keys.",
    ),
    "TOML_PARSE_ERROR": (
        (
            "toml parse error",
            "invalid toml",
            "tomldecodeerror",
            "expected newline",
            "key is not closed",
        ),
        Severity.MEDIUM,
        "A TOML configuration file failed to parse.",
        "Check for missing quotes, invalid keys, or malformed inline tables.",
    ),
    # ── Database ──────────────────────────────────────────────────────
    "DB_CONNECTION_FAILED": (
        (
            "could not connect to server",
            "connection refused",
            "database is locked",
            "unable to connect",
            "cannot connect to database",
            "could not connect to database",
        ),
        Severity.HIGH,
        "A database connection could not be established.",
        "Verify the database host, port, credentials, and that the service is running.",
    ),
    # ── Encoding ──────────────────────────────────────────────────────
    "ENCODING_ERROR": (
        (
            "unicodeencodeerror",
            "unicodedecodeerror",
            "codec can't decode",
            "codec can't encode",
            "invalid byte",
            "invalid utf-8",
        ),
        Severity.MEDIUM,
        "A text encoding or decoding operation failed.",
        "Specify the correct encoding or handle binary data with errors='replace'.",
    ),
    # ── Process ───────────────────────────────────────────────────────
    "PROCESS_KILLED": (
        (
            "signal: killed",
            "signal: terminated",
            "process was killed",
            "killed by signal",
            "sigterm",
            "sighup",
        ),
        Severity.MEDIUM,
        "The process was terminated by an external signal.",
        "Check for OOM killer, systemd limits, or manual kill signals.",
    ),
    # ── Lock File ─────────────────────────────────────────────────────
    "LOCK_CONFLICT": (
        (
            "unable to acquire lock",
            "lock file exists",
            "already locked",
            "another process is running",
            "lock timeout",
        ),
        Severity.MEDIUM,
        "A lock file or mutex prevented the operation from proceeding.",
        "Remove stale lock files or wait for the holding process to finish.",
    ),
    # ── Path / CWD ────────────────────────────────────────────────────
    "PATH_RESOLUTION_ERROR": (
        (
            "not a git repository",
            "not a valid working directory",
            "outside of a project",
            "cannot find project root",
            "no such directory",
        ),
        Severity.MEDIUM,
        "The current working directory or project root is not as expected.",
        "Change to the correct project directory or verify the project structure.",
    ),
    # ── pip / PyPI ────────────────────────────────────────────────────
    "PIP_NO_MATCHING_DIST": (
        (
            "no matching distribution found",
            "could not find a version that satisfies the requirement",
            "no matching distribution",
        ),
        Severity.HIGH,
        "pip could not resolve a compatible package version.",
        "Check the package name, Python version compatibility, and index URL (pip install -i or PIP_INDEX_URL).",
    ),
    "PIP_RESOLUTION_CONFLICT": (
        (
            "resolutionimpossible",
            "pip's dependency resolver does not currently take into account",
            "cannot install because these package versions have conflicting dependencies",
        ),
        Severity.HIGH,
        "pip's resolver found mutually incompatible version constraints.",
        "Relax pinned versions, upgrade pip, or install into a fresh virtual environment.",
    ),
    "PIP_BUILD_FAILED": (
        (
            "failed building wheel for",
            "error: subprocess-exited-with-error",
            "command errored out with exit status",
            "legacy-install-failure",
        ),
        Severity.HIGH,
        "pip failed to build a package from source.",
        "Install build tooling (compilers, headers), or prefer a prebuilt wheel / binary distribution of the package.",
    ),
    # ── Node / npm ────────────────────────────────────────────────────
    "NPM_RESOLVE_ERROR": (
        (
            "npm err! eresolve",
            "unable to resolve dependency tree",
            "could not resolve dependency",
            "fix the upstream dependency conflict",
        ),
        Severity.HIGH,
        "npm could not reconcile peer dependency constraints.",
        "Inspect the conflicting peer ranges; as a last resort use "
        "--legacy-peer-deps or update the conflicting packages.",
    ),
    "NPM_REGISTRY_ERROR": (
        (
            "npm err! 404",
            "npm err! e401",
            "npm err! e403",
            "npm err! code e401",
            "npm err! code e403",
            "not in the npm registry",
            "unable to authenticate",
        ),
        Severity.HIGH,
        "The npm registry rejected the request (missing package or auth).",
        "Verify the package name, registry URL, and credentials in ~/.npmrc or NPM_TOKEN.",
    ),
    "NODE_VERSION_UNSUPPORTED": (
        (
            "unsupported engine",
            "requires node",
            "requires node version",
            'the engine "node" is incompatible',
            "syntaxerror: unexpected token",  # common when old node parses new syntax
        ),
        Severity.MEDIUM,
        "The installed Node.js version does not satisfy the project's engine range.",
        "Switch to a supported Node version (nvm/n/fnm) or upgrade the runtime.",
    ),
    # ── nginx ─────────────────────────────────────────────────────────
    "NGINX_CONFIG_INVALID": (
        (
            "nginx: [emerg]",
            "invalid number of arguments",
            "unknown directive",
            "directive is not allowed here",
            "test is unsuccessful",
        ),
        Severity.HIGH,
        "The nginx configuration failed validation.",
        "Run nginx -t, fix the reported directive/block, then reload the service.",
    ),
    "NGINX_UPSTREAM_DOWN": (
        (
            "connect() failed (111: connection refused) while connecting to upstream",
            "upstream timed out",
            "no live upstreams",
            "502 bad gateway",
            "504 gateway time-out",
        ),
        Severity.HIGH,
        "nginx could not reach its upstream backend.",
        "Verify the upstream service is listening, and check proxy_pass targets and upstream timeouts.",
    ),
    # ── Kubernetes ────────────────────────────────────────────────────
    "K8S_POD_FAILED": (
        (
            "crashloopbackoff",
            "imagepullbackoff",
            "errimagepull",
            "createcontainerconfigerror",
            "pod failed",
        ),
        Severity.HIGH,
        "A Kubernetes pod failed to start or stay running.",
        "Inspect kubectl describe pod events, image pull secrets, and container logs.",
    ),
    "KUBECONFIG_INVALID": (
        (
            "unable to connect to the server",
            "error loading config file",
            "no configuration has been provided",
            "the server has asked for the client to provide credentials",
        ),
        Severity.HIGH,
        "kubectl could not authenticate to the cluster.",
        "Check KUBECONFIG, context selection (kubectl config use-context), and cluster credentials.",
    ),
    # ── systemd / Linux services ──────────────────────────────────────
    "SYSTEMD_UNIT_FAILED": (
        (
            "unit entered failed state",
            "failed with result 'exit-code'",
            "start request repeated too quickly",
            "job for .service failed",
        ),
        Severity.HIGH,
        "A systemd service unit failed to start or stay up.",
        "Inspect journalctl -u <unit> for the failing command and environment.",
    ),
    # ── Java / JVM ────────────────────────────────────────────────────
    "JVM_HEAP_EXHAUSTED": (
        (
            "java.lang.outofmemoryerror",
            "gc overhead limit exceeded",
            "java heap space",
        ),
        Severity.CRITICAL,
        "The JVM ran out of heap or metaspace.",
        "Raise -Xmx, inspect memory leaks, or reduce batch/heap pressure.",
    ),
    "JAVA_VERSION_MISMATCH": (
        (
            "unsupportedclassversionerror",
            "class file has wrong version",
            "release version",
            "invalid target release",
        ),
        Severity.MEDIUM,
        "The class was compiled for a different Java version than the runtime.",
        "Align JAVA_HOME / javac release level with the deployment runtime.",
    ),
    # ── Rust / cargo ──────────────────────────────────────────────────
    "CARGO_BUILD_FAILED": (
        (
            "error: could not compile",
            "linker `cc` not found",
            "error: linking with",
            "cannot find -l",
        ),
        Severity.HIGH,
        "A cargo/rustc build failed at compile or link time.",
        "Install the missing system toolchain/development libraries, or fix the reported type error.",
    ),
    # ── Go ────────────────────────────────────────────────────────────
    "GO_MODULE_ERROR": (
        (
            "no required module provides package",
            "missing go.sum entry",
            "unrecognized import path",
            "module requires go",
        ),
        Severity.MEDIUM,
        "Go module resolution failed.",
        "Run go mod tidy / go get for the missing module, or upgrade the Go toolchain.",
    ),
}


def _compile_boundary_pattern(phrase: str) -> re.Pattern[str]:
    """Compile a phrase so short tokens match only on token boundaries.

    'oom' must not fire inside 'boom'; '429' must not fire inside '42900'.
    A trailing plural 's' is tolerated ('unexpected token' ~ 'unexpected tokens').
    """
    body = re.escape(phrase)
    if phrase[0].isalnum() or phrase[0] == "_":
        body = r"(?<![\w])" + body
    if phrase[-1].isalnum() or phrase[-1] == "_":
        body = body + (r"(?![\w])" if phrase[-1] == "s" else r"s?(?![\w])")
    return re.compile(body)


_COMPILED_ERROR_PATTERNS: dict[str, tuple[tuple[re.Pattern[str], ...], Severity, str, str]] = {
    code: (tuple(_compile_boundary_pattern(p) for p in patterns), severity, meaning, remediation)
    for code, (patterns, severity, meaning, remediation) in GENERIC_ERROR_PATTERNS.items()
}


def generic_error_evidence(text: str, corrections: dict[str, dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    """Return structured matches from free-form terminal or log text.

    Operator corrections (fingerprint -> override) take precedence over the
    built-in patterns: a previously confirmed label always wins, which is what
    makes the engine "learn" from past mistakes without an LLM.
    """
    corrections = corrections or {}
    matches: list[dict[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        lowered = line.lower()
        fingerprint = normalize_error_text(line)
        override = corrections.get(fingerprint)
        if override is not None:
            matches.append(
                {
                    "code": override["code"],
                    "severity": str(override.get("severity", "MEDIUM")).lower(),
                    "meaning": override.get("meaning", "Operator-confirmed classification."),
                    "remediation": override.get("remediation", "See the recorded correction."),
                    "line": line_number,
                    "text": line.strip()[:500],
                    "source": "correction",
                    "fingerprint": fingerprint,
                }
            )
            continue
        for code, (patterns, severity, meaning, remediation) in _COMPILED_ERROR_PATTERNS.items():
            if any(rx.search(lowered) for rx in patterns):
                matches.append(
                    {
                        "code": code,
                        "severity": severity.value,
                        "meaning": meaning,
                        "remediation": remediation,
                        "line": line_number,
                        "text": line.strip()[:500],
                        "source": "rule",
                    }
                )
    return matches[:50]


def classify_error(
    text: str,
    exit_code: int | None = None,
    corrections: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """High-level classification used by the planning layer."""
    findings = generic_error_evidence(text, corrections)
    codes = {f["code"] for f in findings}
    if exit_code not in (None, 0) and "EXIT_NON_ZERO" not in codes:
        findings.append(
            {
                "code": "EXIT_NON_ZERO",
                "severity": Severity.MEDIUM.value,
                "meaning": "The analyzed command exited with a non-zero status.",
                "remediation": "Inspect the command's stderr/stdout and surrounding context before retrying.",
                "line": 1,
                "text": f"exit_code={exit_code}",
            }
        )
        codes.add("EXIT_NON_ZERO")

    primary = None
    for code in (
        "MEMORY_EXHAUSTED",
        "DISK_FULL",
        "DOCKER_NOT_RUNNING",
        "GIT_AUTH_FAILED",
        "COMMAND_NOT_FOUND",
        "MODULE_NOT_FOUND",
        "NETWORK_FAILURE",
        "PERMISSION_DENIED",
        "SSL_CERT_ERROR",
        "DB_CONNECTION_FAILED",
    ):
        if code in codes:
            primary = code
            break

    return {
        "findings": findings,
        "codes": sorted(codes),
        "primary_code": primary,
        "has_actionable": bool(codes),
    }


def suggest_followup_command(text: str, codes: set[str], language: str, command: str) -> str | None:
    """Suggest a safe, read-only verification command based on the primary finding."""
    lang = language.lower().strip()
    if "MODULE_NOT_FOUND" in codes and lang in {"python", "py", ""}:
        return f'"{sys.executable}" -c "import sys; print(sys.executable)"'
    if "COMMAND_NOT_FOUND" in codes and command:
        first = command.strip().split()[0]
        if first:
            return f"where {first}" if os.name == "nt" else f"which {first}"
    if "FILE_NOT_FOUND" in codes:
        return 'python -c "import os; print(os.getcwd())"'
    if "ENV_VAR_MISSING" in codes:
        return "set" if os.name == "nt" else "env"
    if "NETWORK_FAILURE" in codes:
        return "ping 127.0.0.1"
    if "DOCKER_NOT_RUNNING" in codes:
        return "docker info"
    if "DOCKER_IMAGE_NOT_FOUND" in codes:
        return "docker images"
    if "GIT_AUTH_FAILED" in codes:
        return "git remote -v"
    if "GIT_MERGE_CONFLICT" in codes:
        return "git status"
    if "DISK_FULL" in codes:
        if os.name == "nt":
            return 'powershell -NoProfile -Command "Get-CimInstance Win32_LogicalDisk | Select-Object DeviceID,Size,FreeSpace"'
        return "df -h"
    if "PORT_IN_USE" in codes:
        return "netstat -ano" if os.name == "nt" else "ss -tlnp"
    if "TIMEOUT_ERROR" in codes:
        return "ping 8.8.8.8"
    if "DB_CONNECTION_FAILED" in codes:
        return "netstat -ano | findstr :5432" if os.name == "nt" else "ss -tlnp | grep :5432"
    if "PATH_RESOLUTION_ERROR" in codes:
        return "cd" if os.name == "nt" else "pwd"
    if "LOCK_CONFLICT" in codes:
        return "dir *.lock" if os.name == "nt" else "ls -la *.lock 2>/dev/null; echo 'No lock files'"
    return None


_HIGH_RISK_COMMAND_RE = re.compile(
    r"\b(rm\s+-[a-z]*[rf]|del\s+/[fsq]|erase\s+/|format\b|mkfs|dd\s+if=|shutdown|reboot|poweroff|halt"
    r"|diskpart|reg\s+(delete|add)|remove-item\b.*-recurse|rd\s+/s|rmdir\s+/s"
    r"|git\s+push\b.*--force|git\s+reset\s+--hard|drop\s+(table|database)|truncate\s+table"
    r"|kubectl\s+delete|helm\s+(uninstall|delete)|terraform\s+destroy|chmod\s+-r\s+777)",
    re.IGNORECASE,
)

_READ_ONLY_PREFIXES = (
    "which", "where", "pwd", "cd", "echo", "env", "set", "ls", "dir", "df", "du",
    "ping", "netstat", "ss", "ip", "ifconfig", "docker info", "docker images", "docker ps",
    "git status", "git remote", "git log", "git diff", "kubectl get", "kubectl describe",
    "systemctl status", "journalctl", "python -c", "pip list", "pip show", "npm list",
    "powershell", "cat", "type", "get-content",
)


def classify_command_risk(command: str) -> str:
    """Heuristic risk label for a proposed shell command: 'low' | 'medium' | 'high'."""
    text = command.strip().lower()
    if not text:
        return "medium"
    if _HIGH_RISK_COMMAND_RE.search(text):
        return "high"
    if any(text == p or text.startswith(p + " ") for p in _READ_ONLY_PREFIXES):
        return "low"
    return "medium"