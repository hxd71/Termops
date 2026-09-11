from __future__ import annotations

import time

import pytest
from fastapi.testclient import TestClient

from termops.api import create_app
from termops.engine import OpsEngine
from termops.models import TargetRef, TaskKind, TaskStatus
from termops.web import stream_task_events


def auth(engine: OpsEngine) -> dict[str, str]:
    return {"X-Operator-Token": engine.operator_token}


def test_probe_accepts_empty_body(settings) -> None:
    engine = OpsEngine(settings)
    with TestClient(create_app(settings, engine)) as client:
        response = client.post("/v1/tasks/probe", json={}, headers=auth(engine))
        assert response.status_code == 202
        assert response.json()["kind"] == "probe"
    engine.store.close()


def test_api_requires_operator_token(settings) -> None:
    engine = OpsEngine(settings)
    with TestClient(create_app(settings, engine)) as client:
        assert client.get("/v1/capabilities").status_code == 401
        assert client.get("/v1/capabilities", headers=auth(engine)).status_code == 200
    engine.store.close()


def test_generic_analysis_submission(settings) -> None:
    engine = OpsEngine(settings)
    with TestClient(create_app(settings, engine)) as client:
        response = client.post(
            "/v1/tasks/analyze",
            headers=auth(engine),
            json={
                "text": "ModuleNotFoundError: No module named 'requests'\n",
                "source": "stderr",
                "language": "python",
                "command": "pytest -q",
                "cwd": "/workspace/app",
                "exit_code": 1,
            },
        )
        assert response.status_code == 202
        task_id = response.json()["id"]
        for _ in range(100):
            detail = client.get(f"/v1/tasks/{task_id}", headers=auth(engine)).json()
            if detail["task"]["status"] == "waiting_approval":
                break
            time.sleep(0.01)
        assert detail["task"]["status"] == "waiting_approval"
        assert detail["actions"]
        assert detail["actions"][0]["kind"] == "run_command"
    engine.store.close()


def test_generic_analysis_without_command_completes(settings) -> None:
    engine = OpsEngine(settings)
    with TestClient(create_app(settings, engine)) as client:
        response = client.post(
            "/v1/tasks/analyze",
            headers=auth(engine),
            json={
                "text": "ModuleNotFoundError: No module named 'requests'\n",
                "source": "stderr",
                "language": "python",
                "command": "",
                "cwd": "/workspace/app",
                "exit_code": 1,
            },
        )
        assert response.status_code == 202
        task_id = response.json()["id"]
        for _ in range(100):
            detail = client.get(f"/v1/tasks/{task_id}", headers=auth(engine)).json()
            if detail["task"]["status"] == "waiting_approval":
                break
            time.sleep(0.01)
        assert detail["task"]["status"] == "waiting_approval"
        action = detail["actions"][0]
        client.post(
            f"/v1/actions/{action['id']}/decision",
            headers=auth(engine),
            json={"decision": "approve", "action_digest": action["digest"]},
        )
        for _ in range(100):
            detail = client.get(f"/v1/tasks/{task_id}", headers=auth(engine)).json()
            if detail["task"]["status"] == "succeeded":
                break
            time.sleep(0.01)
        assert detail["task"]["status"] == "succeeded"
        task_knowledge = client.get(f"/v1/tasks/{task_id}/knowledge", headers=auth(engine)).json()
        assert task_knowledge
        global_knowledge = client.get("/v1/knowledge", headers=auth(engine)).json()
        assert any(item["task_id"] == task_id for item in global_knowledge)
    engine.store.close()


def test_web_login_diagnosis_and_csrf(settings) -> None:
    engine = OpsEngine(settings)
    app = create_app(settings, engine)
    with TestClient(app) as client:
        # Submit an analysis task
        response = client.post(
            "/v1/tasks/analyze",
            headers=auth(engine),
            json={
                "text": "ModuleNotFoundError: No module named 'click'\n",
                "source": "stderr",
                "language": "python",
            },
        )
        assert response.status_code == 202
        task_id = response.json()["id"]
        for _ in range(100):
            detail = client.get(f"/v1/tasks/{task_id}", headers=auth(engine)).json()
            if detail["task"]["status"] in {"succeeded", "waiting_approval"}:
                break
            time.sleep(0.01)

        # Login via web
        login = client.post("/v1/web/login-code", headers=auth(engine)).json()
        code = login["url"].split("code=", 1)[1]
        logged_in = client.get(f"/login?code={code}", follow_redirects=True)
        assert logged_in.status_code == 200
        assert "运行总览" in logged_in.text
        csrf = logged_in.text.split('name="csrf-token" content="', 1)[1].split('"', 1)[0]

        # CSRF is required for web API
        assert client.post("/ui-api/analyze", json={"text": "x"}).status_code == 403

        # Analyze via web API with CSRF
        analyzed = client.post(
            "/ui-api/analyze",
            headers={"X-CSRF-Token": csrf},
            json={"text": "ModuleNotFoundError: No module named 'x'", "language": "python"},
        )
        assert analyzed.status_code == 202

        # View task page
        task_page = client.get(analyzed.json()["location"])
        assert task_page.status_code == 200
        assert "Findings" in task_page.text or "findings" in task_page.text.lower()
    engine.store.close()


def test_corrections_api_roundtrip(settings) -> None:
    engine = OpsEngine(settings)
    app = create_app(settings, engine)
    with TestClient(app) as client:
        # Invalid severity is rejected by the schema.
        bad = client.post(
            "/v1/corrections",
            headers=auth(engine),
            json={"sample_text": "boom", "code": "X", "severity": "SEVERE", "meaning": "m"},
        )
        assert bad.status_code == 422

        created = client.post(
            "/v1/corrections",
            headers=auth(engine),
            json={
                "sample_text": "ERROR: disk quota exceeded on /var/lib/app",
                "code": "DISK_QUOTA_EXCEEDED",
                "severity": "HIGH",
                "meaning": "Per-user quota reached.",
                "remediation": "Clean up or raise the quota.",
            },
        )
        assert created.status_code == 201
        fingerprint = created.json()["fingerprint"]

        listed = client.get("/v1/corrections", headers=auth(engine)).json()
        assert any(row["fingerprint"] == fingerprint for row in listed)

        deleted = client.delete(f"/v1/corrections/{fingerprint}", headers=auth(engine)).json()
        assert deleted["deleted"] is True
    engine.store.close()


def test_web_login_lockout_and_logout(settings) -> None:
    engine = OpsEngine(settings)
    app = create_app(settings, engine)
    with TestClient(app) as client:
        # Repeated bad codes from the same client eventually lock out (429).
        for _ in range(5):
            bad = client.get("/login?code=definitely-wrong")
        assert bad.status_code in {401, 429}
        assert client.get("/login?code=definitely-wrong").status_code == 429

    # A fresh app instance resets the limiter (in-memory by design).
    with TestClient(create_app(settings, engine)) as client:
        # Valid login still works, then logout invalidates the session.
        login = client.post("/v1/web/login-code", headers=auth(engine)).json()
        code = login["url"].split("code=", 1)[1]
        logged_in = client.get(f"/login?code={code}", follow_redirects=True)
        assert logged_in.status_code == 200

        client.get("/logout", follow_redirects=False)
        # Session cookie is gone; the UI redirects back to /login.
        after = client.get("/ui/", follow_redirects=False)
        assert after.status_code == 303
        assert after.headers["location"] == "/login"
    engine.store.close()


def test_cancel_invalidates_pending_action(settings) -> None:
    engine = OpsEngine(settings)
    with TestClient(create_app(settings, engine)) as client:
        created = client.post(
            "/v1/tasks/analyze",
            headers=auth(engine),
            json={
                "text": "ModuleNotFoundError: No module named 'requests'\n",
                "source": "stderr",
                "language": "python",
                "command": "python -c \"print('test')\"",
                "exit_code": 1,
            },
        ).json()
        for _ in range(100):
            detail = client.get(f"/v1/tasks/{created['id']}", headers=auth(engine)).json()
            if detail["task"]["status"] == "waiting_approval":
                break
            time.sleep(0.01)
        action = detail["actions"][0]

        cancelled = client.post(f"/v1/tasks/{created['id']}/cancel", headers=auth(engine))
        assert cancelled.status_code == 200
        assert cancelled.json()["status"] == "cancelled"
        assert engine.store.get_action(action["id"]).status.value == "rejected"

        replay = client.post(
            f"/v1/actions/{action['id']}/decision",
            headers=auth(engine),
            json={"decision": "approve", "action_digest": action["digest"]},
        )
        assert replay.status_code == 422
        assert "not waiting for approval" in replay.json()["detail"]
    engine.store.close()


class ConnectedOnce:
    def __init__(self) -> None:
        self.checks = 0

    async def is_disconnected(self) -> bool:
        self.checks += 1
        return self.checks > 1


@pytest.mark.asyncio
async def test_sse_stream_emits_auditable_task_updates(settings) -> None:
    engine = OpsEngine(settings)
    task = engine.store.create_task(
        kind=TaskKind.ANALYZE,
        target=TargetRef(kind="workspace", name="test"),
        input_data={},
    )
    engine.store.update_task(task.id, TaskStatus.RUNNING)

    chunks = [chunk async for chunk in stream_task_events(ConnectedOnce(), engine.store, task.id, poll_seconds=0)]

    payload = "".join(chunks)
    assert "event: audit" in payload
    assert '"event_type": "task.status"' in payload
    assert '"to": "running"' in payload
    assert payload.endswith(": keepalive\n\n")
    engine.store.close()