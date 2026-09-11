"""Deterministic error-classification engine tests.

The production core is LLM-free by design, so these tests pin the exact error
codes, severities, and follow-up commands the classifier emits.
"""

from __future__ import annotations

import os

import pytest

from termops.diagnostics import classify_error, generic_error_evidence, suggest_followup_command


@pytest.mark.parametrize(
    ("text", "expected_code"),
    [
        ("PermissionError: [Errno 13] Permission denied", "PERMISSION_DENIED"),
        ("OSError: [Errno 98] Address already in use", "PORT_IN_USE"),
        ("psycopg2.OperationalError: database is locked", "DB_CONNECTION_FAILED"),
        ("OSError: [Errno 28] No space left on device", "DISK_FULL"),
        ("httpx.ReadTimeout: timed out", "TIMEOUT_ERROR"),
        ("openai.RateLimitError: 429 Too Many Requests", "RATE_LIMITED"),
        ("ssl.SSLCertVerificationError: certificate verify failed", "SSL_CERT_ERROR"),
        ("UnicodeDecodeError: codec can't decode byte 0xff", "ENCODING_ERROR"),
        ("fatal: not a git repository", "PATH_RESOLUTION_ERROR"),
        ("ERROR: lock file exists", "LOCK_CONFLICT"),
        ("error: failed to push some refs", "GIT_REMOTE_REJECTED"),
        ("RecursionError: maximum recursion depth exceeded", "STACK_OVERFLOW"),
        ("NodeError: Cannot find module 'express'", "MODULE_NOT_FOUND"),
        ("docker: Cannot connect to the Docker daemon", "DOCKER_NOT_RUNNING"),
        ("npm ERR! package was not found", "PACKAGE_NOT_FOUND"),
        ("VerificationError: version conflict", "DEPENDENCY_CONFLICT"),
        ("MemoryError: out of memory", "MEMORY_EXHAUSTED"),
    ],
)
def test_classify_error_recognizes_pattern(text: str, expected_code: str) -> None:
    result = classify_error(text)
    assert expected_code in result["codes"]


def test_classify_error_empty_input() -> None:
    result = classify_error("")
    assert result["findings"] == []
    assert result["codes"] == []
    assert result["primary_code"] is None
    assert result["has_actionable"] is False


def test_classify_error_adds_exit_code_finding() -> None:
    result = classify_error("some benign text", exit_code=1)
    assert "EXIT_NON_ZERO" in result["codes"]
    assert result["has_actionable"] is True


def test_generic_error_evidence_is_structured() -> None:
    matches = generic_error_evidence("PermissionError: [Errno 13] Permission denied")
    assert matches
    match = matches[0]
    assert match["code"] == "PERMISSION_DENIED"
    assert match["severity"] == "high"
    assert match["line"] == 1
    assert "remediation" in match


def test_generic_error_evidence_no_match() -> None:
    assert generic_error_evidence("all good, nothing to see here") == []


def test_normalize_error_text_produces_stable_fingerprints() -> None:
    from termops.diagnostics import normalize_error_text

    # Two occurrences of the same error with different volatile details
    # must normalize to the same key — that is the whole feedback-loop premise.
    a = normalize_error_text(r"Permission denied: C:\Users\alice\app\config.json (line 42)")
    b = normalize_error_text(r"Permission denied: C:\Users\bob\service\settings.toml (line 7)")
    assert a == b
    assert r"C:\Users" not in a
    assert "line" in a

    c = normalize_error_text("Connection refused to 10.0.0.1:8080")
    d = normalize_error_text("Connection refused to 192.168.1.50:5432")
    assert c == d
    assert "10.0.0.1" not in c

    # Distinct errors keep distinct fingerprints.
    assert a != c


def test_corrections_override_builtin_rules() -> None:
    from termops.diagnostics import classify_error, normalize_error_text

    sample = "Permission denied: /var/lib/app/data.db"
    correction = {
        "code": "APP_DATA_LOCKED",
        "severity": "HIGH",
        "meaning": "The app data file is locked by another process.",
        "remediation": "Stop the other instance and retry.",
    }
    corrections = {normalize_error_text(sample): correction}
    result = classify_error(sample, corrections=corrections)
    assert "APP_DATA_LOCKED" in result["codes"]
    finding = next(f for f in result["findings"] if f["code"] == "APP_DATA_LOCKED")
    assert finding["meaning"] == correction["meaning"]


def test_new_tool_specific_patterns_classify() -> None:
    from termops.diagnostics import generic_error_evidence

    cases = [
        # Codes asserted against the actual pattern catalog in diagnostics.py.
        ("npm ERR! 404 Not Found - GET https://registry.npmjs.org/xyz", "NPM_REGISTRY_ERROR"),
        ("npm ERR! ERESOLVE unable to resolve dependency tree", "NPM_RESOLVE_ERROR"),
        (
            "Could not find a version that satisfies the requirement tensorflow",
            "PIP_NO_MATCHING_DIST",
        ),
        ("failed building wheel for numpy", "PIP_BUILD_FAILED"),
        ("nginx: [emerg] unknown directive 'proxypass'", "NGINX_CONFIG_INVALID"),
        ("CrashLoopBackOff: back-off 5m0s restarting failed container", "K8S_POD_FAILED"),
        ("java.lang.OutOfMemoryError: Java heap space", "JVM_HEAP_EXHAUSTED"),
        ("error: could not compile `myapp` due to previous error", "CARGO_BUILD_FAILED"),
        ("no required module provides package example.com/foo", "GO_MODULE_ERROR"),
        (
            "Failed to start myapp.service: Unit entered failed state.",
            "SYSTEMD_UNIT_FAILED",
        ),
    ]
    for text, expected in cases:
        codes = {m["code"] for m in generic_error_evidence(text)}
        assert expected in codes, f"{expected} not found for: {text!r} (got {codes})"


def test_classify_error_primary_code_priority() -> None:
    result = classify_error("connection refused")
    assert result["primary_code"] == "NETWORK_FAILURE"
    assert "DB_CONNECTION_FAILED" in result["codes"]


def test_suggest_followup_module_not_found() -> None:
    cmd = suggest_followup_command("no module named requests", {"MODULE_NOT_FOUND"}, "python", "")
    assert cmd is not None and "import sys" in cmd


def test_suggest_followup_command_not_found() -> None:
    cmd = suggest_followup_command("'git' is not recognized", {"COMMAND_NOT_FOUND"}, "", "git status")
    assert cmd == ("where git" if os.name == "nt" else "which git")


def test_suggest_followup_network() -> None:
    assert suggest_followup_command("connection refused", {"NETWORK_FAILURE"}, "", "") == "ping 127.0.0.1"


def test_suggest_followup_disk_full() -> None:
    expected = (
        'powershell -NoProfile -Command "Get-CimInstance Win32_LogicalDisk | Select-Object DeviceID,Size,FreeSpace"'
        if os.name == "nt"
        else "df -h"
    )
    assert suggest_followup_command("no space left", {"DISK_FULL"}, "", "") == expected


def test_noisy_substrings_do_not_false_positive() -> None:
    # 'oom' inside 'boom'/'zoom' must not match MEMORY_EXHAUSTED
    assert "MEMORY_EXHAUSTED" not in {m["code"] for m in generic_error_evidence("the boom microphone zoom failed")}
    # '429' inside a port number must not match RATE_LIMITED
    assert "RATE_LIMITED" not in {m["code"] for m in generic_error_evidence("listening on port 42900 ok")}
    # a yaml filename alone is not a yaml parse error
    assert "CONFIG_PARSE_ERROR" not in {m["code"] for m in generic_error_evidence("reading config.yaml failed: permission denied")}
    # 'expected'/'actual' as plain words must not match TEST_FAILURE
    assert "TEST_FAILURE" not in {m["code"] for m in generic_error_evidence("the expected actual behavior differs")}


def test_suggest_followup_returns_none_for_unknown_codes() -> None:
    assert suggest_followup_command("x", {"SYNTAX_ERROR"}, "", "") is None