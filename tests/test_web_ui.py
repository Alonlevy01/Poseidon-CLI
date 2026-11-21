"""Unit and API tests for Phase D: Poseidon Web UI (Starlette + WebSockets + /commands)."""

from pathlib import Path

from starlette.testclient import TestClient

from poseidon.config import Settings
from poseidon.web.server import create_web_app


def test_web_ui_serves_html_state_and_enforces_workspace_security(tmp_path: Path) -> None:
    (tmp_path / "ledger.py").write_text("print('ok')\n", encoding="utf-8")
    (tmp_path / ".env").write_text("OPENROUTER_API_KEY=secret\n", encoding="utf-8")

    app = create_web_app(workspace=tmp_path, settings=Settings())
    client = TestClient(app)

    # 1. Serves Abyssal Ocean Web UI with custom trident symbol and /ws endpoint
    home = client.get("/")
    assert home.status_code == 200
    assert "POSEIDON" in home.text
    assert "poseidon-trident" in home.text
    assert "/ws" in home.text

    # 2. State endpoint lists clean files, slash commands, and excludes .env
    st = client.get("/api/state")
    assert st.status_code == 200
    payload = st.json()
    assert payload["files"] == ["ledger.py"]
    assert payload["plan_mode"] is False
    assert payload["mode"] == "agent"
    assert len(payload["slash_commands"]) >= 20

    # 3. File viewer allows workspace file and blocks secrets & path traversal
    ok_file = client.get("/api/file", params={"path": "ledger.py"})
    assert ok_file.status_code == 200
    assert "print('ok')" in ok_file.json()["content"]

    assert client.get("/api/file", params={"path": ".env"}).status_code == 403
    assert client.get("/api/file", params={"path": "../outside.py"}).status_code == 403

    # 4. Tools endpoint lists active built-in tools
    tools_resp = client.get("/api/tools")
    assert tools_resp.status_code == 200
    tool_names = {t["name"] for t in tools_resp.json()["tools"]}
    assert "read_file" in tool_names
    assert "apply_diff" in tool_names

    # 5. Settings, compact, undo, and WebSocket handshake work cleanly
    updated = client.post(
        "/api/settings",
        json={"plan_mode": True, "auto_approve": True, "max_calls": 12, "mode": "diff"},
    )
    assert updated.status_code == 200
    assert updated.json()["plan_mode"] is True
    assert updated.json()["max_calls"] == 12
    assert updated.json()["mode"] == "diff"

    compacted = client.post("/api/compact")
    assert compacted.status_code == 200
    assert compacted.json()["stats"]["masked_count"] == 0

    undone = client.post("/api/undo")
    assert undone.status_code == 200
    assert undone.json()["undone"] is False


def test_web_ui_slash_commands_have_full_terminal_parity(tmp_path: Path) -> None:
    (tmp_path / "analytics.py").write_text(
        "def compute_total(items):\n    return sum(items)\n",
        encoding="utf-8",
    )
    app = create_web_app(workspace=tmp_path, settings=Settings())
    client = TestClient(app)

    # /help, /init, /memory, /plan, /tools, /files, /grep, /limit, /timeout, /auto, /mode
    help_res = client.post("/api/command", json={"command": "/help"}).json()
    assert help_res["ok"] is True
    assert len(help_res["rows"]) >= 20

    init_res = client.post("/api/command", json={"command": "/init"}).json()
    assert init_res["ok"] is True
    assert (tmp_path / "POSEIDON.md").exists()

    mem_res = client.post("/api/command", json={"command": "/memory"}).json()
    assert "POSEIDON.md" in mem_res["title"]

    plan_res = client.post("/api/command", json={"command": "/plan"}).json()
    assert plan_res["state"]["plan_mode"] is True

    tools_plan = client.post("/api/command", json={"command": "/tools"}).json()
    plan_tool_names = {r[0] for r in tools_plan["rows"]}
    assert "apply_diff" not in plan_tool_names  # physically removed in Plan Mode!

    grep_res = client.post("/api/command", json={"command": "/grep compute_total"}).json()
    assert grep_res["ok"] is True
    assert len(grep_res["hits"]) == 1
    assert grep_res["hits"][0]["path"] == "analytics.py"

    limit_res = client.post("/api/command", json={"command": "/limit 11"}).json()
    assert limit_res["state"]["max_calls"] == 11
    timeout_res = client.post("/api/command", json={"command": "/timeout 25"}).json()
    assert timeout_res["state"]["test_timeout"] == 25
    auto_res = client.post("/api/command", json={"command": "/auto on"}).json()
    assert auto_res["state"]["auto_approve"] is True
    mode_res = client.post("/api/command", json={"command": "/mode whole"}).json()
    assert mode_res["state"]["mode"] == "whole"

    # /save, /sessions, /clear, /resume via WebSocket!
    with client.websocket_connect("/ws") as ws:
        hello = ws.receive_json()
        assert hello["type"] == "connected"

        ws.send_json({"type": "prompt", "prompt": "/save demo-web"})
        saved_ev = ws.receive_json()
        assert saved_ev["type"] == "command_result"
        assert saved_ev["result"]["ok"] is True

        ws.send_json({"type": "prompt", "prompt": "/sessions"})
        sess_ev = ws.receive_json()
        assert any(r[0] == "demo-web" for r in sess_ev["result"]["rows"])

        ws.send_json({"type": "prompt", "prompt": "/clear"})
        clear_ev = ws.receive_json()
        assert clear_ev["result"]["clear_stream"] is True

        ws.send_json({"type": "prompt", "prompt": "/resume demo-web"})
        resumed_ev = ws.receive_json()
        assert resumed_ev["result"]["ok"] is True
        assert resumed_ev["result"]["state"]["session_id"] == "demo-web"


def test_web_ui_shutdown_endpoint_and_exit_command_trigger_clean_stop(tmp_path: Path) -> None:
    import threading

    stopped = threading.Event()
    app = create_web_app(
        workspace=tmp_path,
        settings=Settings(),
        on_shutdown=stopped.set,
    )
    client = TestClient(app)

    res = client.post("/api/command", json={"command": "/exit"})
    assert res.status_code == 200
    assert res.json()["shutdown"] is True
    assert stopped.wait(timeout=1.5) is True

    stopped2 = threading.Event()
    app2 = create_web_app(
        workspace=tmp_path,
        settings=Settings(),
        on_shutdown=stopped2.set,
    )
    client2 = TestClient(app2)
    res2 = client2.post("/api/shutdown")
    assert res2.status_code == 200
    assert res2.json()["ok"] is True
    assert stopped2.wait(timeout=1.5) is True


def test_web_ui_resumes_terminal_session_on_connect(tmp_path: Path) -> None:
    from poseidon.context.sessions import SessionStore

    store = SessionStore(tmp_path)
    store.save(
        session_id="cli-handoff",
        model="nvidia/nemotron-3.5-super-vl:free",
        messages=[
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "Hello from terminal"},
            {"role": "assistant", "content": "Hi from Poseidon!"},
        ],
        plan_mode=True,
    )

    app = create_web_app(
        workspace=tmp_path,
        settings=Settings(),
        resume_session_id="cli-handoff",
    )
    client = TestClient(app)
    with client.websocket_connect("/ws") as ws:
        hello = ws.receive_json()
        assert hello["type"] == "connected"
        assert hello["state"]["session_id"] == "cli-handoff"
        assert hello["state"]["plan_mode"] is True
        assert len(hello["messages"]) == 2
        assert hello["messages"][0]["content"] == "Hello from terminal"

    # Deleting the saved session removes the JSONL file and updates state
    del_resp = client.post("/api/sessions/delete", json={"session_id": "cli-handoff"})
    assert del_resp.status_code == 200
    assert del_resp.json()["deleted"] is True
    assert del_resp.json()["cleared_active"] is True
    assert del_resp.json()["state"]["sessions"] == []

    # Deleting again returns 404
    missing_resp = client.post("/api/sessions/delete", json={"session_id": "cli-handoff"})
    assert missing_resp.status_code == 404



