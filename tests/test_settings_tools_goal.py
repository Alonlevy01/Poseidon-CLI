"""Unit tests for persistent Settings, MCP CRUD, Web/Git tools, and `/goal` autonomous loop."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from starlette.testclient import TestClient

from poseidon.config import Settings
from poseidon.context.user_settings import (
    UserSettingsStore,
    add_mcp_server,
    list_mcp_configs,
    remove_mcp_server,
)
from poseidon.core.goal import parse_goal_command, run_goal_loop
from poseidon.llm.openrouter import ChatResponse, TokenUsage, ToolCall
from poseidon.sandbox.docker_runner import TestRunResult
from poseidon.tools.builtin import register_core_tools
from poseidon.tools.registry import ToolRegistry, ToolSessionState
from poseidon.tools.web_tools import _validate_public_url, html_to_clean_text
from poseidon.web.server import create_web_app
from poseidon.workspace import WorkspaceGuard


def test_user_settings_and_mcp_crud_via_api_and_commands(tmp_path: Path) -> None:
    store = UserSettingsStore(tmp_path)
    store.add_custom_model("qwen/qwen3-coder:free")
    assert "qwen/qwen3-coder:free" in store.load().custom_models

    with pytest.raises(ValueError):
        store.add_custom_model("not-a-valid-slug")

    add_mcp_server(tmp_path, name="tax", command="python", args=["/workspace/tax.py"])
    assert "tax" in list_mcp_configs(tmp_path)
    removed, _ = remove_mcp_server(tmp_path, "tax")
    assert removed is True
    assert list_mcp_configs(tmp_path) == {}

    # Web API parity: Settings + Custom Models + MCP + /settings + /mcp + /web
    app = create_web_app(workspace=tmp_path, settings=Settings())
    client = TestClient(app)

    add_m = client.post(
        "/api/models/add",
        json={"model_id": "deepseek/deepseek-r1:free", "activate": True},
    )
    assert add_m.status_code == 200
    assert add_m.json()["state"]["model"] == "deepseek/deepseek-r1:free"
    assert "deepseek/deepseek-r1:free" in add_m.json()["state"]["custom_models"]

    rm_m = client.post("/api/models/remove", json={"model_id": "deepseek/deepseek-r1:free"})
    assert rm_m.status_code == 200
    assert "deepseek/deepseek-r1:free" not in rm_m.json()["state"]["custom_models"]

    add_mcp = client.post(
        "/api/mcp/add",
        json={"name": "docs", "command": "python", "args": "/workspace/docs_mcp.py"},
    )
    assert add_mcp.status_code == 200
    assert add_mcp.json()["state"]["mcp_server_count"] == 1

    set_cmd = client.post("/api/command", json={"command": "/settings"}).json()
    assert set_cmd["ok"] is True
    assert any(r[0] == "Web Search & Fetch Tools" for r in set_cmd["rows"])

    web_off = client.post("/api/command", json={"command": "/web off"}).json()
    assert web_off["state"]["web_tools_enabled"] is False
    tools_off = client.get("/api/tools").json()["tools"]
    assert "web_search" not in {t["name"] for t in tools_off}

    rm_mcp = client.post("/api/mcp/remove", json={"name": "docs"})
    assert rm_mcp.status_code == 200
    assert rm_mcp.json()["state"]["mcp_server_count"] == 0


def test_web_tools_ssrf_guard_and_html_cleaner() -> None:
    for bad in (
        "http://localhost:2701/api/state",
        "http://127.0.0.1/secret",
        "http://169.254.169.254/latest/meta-data/",
        "file:///etc/passwd",
        "ftp://example.com/file",
    ):
        with pytest.raises(ValueError):
            _validate_public_url(bad)

    cleaned = html_to_clean_text(
        "<html><head><style>body{color:red}</style><script>alert(1)</script></head>"
        "<body><h1>FastAPI</h1><p>Async &amp; fast.</p></body></html>"
    )
    assert "FastAPI" in cleaned
    assert "Async & fast." in cleaned
    assert "alert" not in cleaned
    assert "color:red" not in cleaned


class _FakeGoalClient:
    """Deterministic fake LLM client driving Plan -> Execute -> Verify in `/goal` tests."""

    def __init__(self, responses: list[ChatResponse]) -> None:
        self._responses = list(responses)
        self.calls = 0

    def chat_completion(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.0,
        **_kwargs: Any,
    ) -> ChatResponse:
        self.calls += 1
        return self._responses.pop(0)


class _FlakyThenPassingRunner:
    def __init__(self) -> None:
        self.runs = 0

    def run(self, workspace_root: Path) -> TestRunResult:
        self.runs += 1
        if self.runs == 1:
            return TestRunResult(
                command="pytest",
                exit_code=1,
                timeout_seconds=15,
                elapsed_seconds=0.1,
                passed=0,
                failed=1,
                errors=0,
                total=1,
                verified=False,
                unverified_reason="1 test failed",
            )
        return TestRunResult(
            command="pytest",
            exit_code=0,
            timeout_seconds=15,
            elapsed_seconds=0.1,
            passed=1,
            failed=0,
            errors=0,
            total=1,
            verified=True,
        )


def test_goal_loop_plans_executes_enforces_verification_and_completes(tmp_path: Path) -> None:
    obj, steps = parse_goal_command("--steps 4 Add greeting helper", default_steps=5)
    assert obj == "Add greeting helper"
    assert steps == 4

    guard = WorkspaceGuard(tmp_path)
    (tmp_path / "app.py").write_text("x = 1\n", encoding="utf-8")
    state = ToolSessionState()
    runner = _FlakyThenPassingRunner()

    ro_reg = ToolRegistry()
    rw_reg = ToolRegistry()
    register_core_tools(
        registry=ro_reg,
        workspace=guard,
        state=state,
        read_only=True,
        enable_web_tools=False,
    )
    register_core_tools(
        registry=rw_reg,
        workspace=guard,
        state=state,
        confirm_patch=lambda _: True,
        test_runner=runner,
        read_only=False,
        enable_web_tools=False,
    )

    fake_client = _FakeGoalClient(
        [
            # Phase 1: Plan response
            ChatResponse(
                content="1. Create greet.py\n2. Run tests",
                model="fake/model",
                finish_reason="stop",
                elapsed_seconds=0.1,
                usage=TokenUsage(total_tokens=20),
            ),
            # Iteration 1: create_file -> auto_verify fails (runner #1) -> agent repairs -> passes!
            ChatResponse(
                content="",
                model="fake/model",
                finish_reason="tool_calls",
                elapsed_seconds=0.1,
                tool_calls=[
                    ToolCall(
                        id="c1",
                        name="create_file",
                        arguments='{"path": "greet.py", "content": "def hi():\\n    return 1\\n"}',
                    )
                ],
                usage=TokenUsage(total_tokens=20),
            ),
            ChatResponse(
                content="Created greet.py\n<goal_status>COMPLETE</goal_status>",
                model="fake/model",
                finish_reason="stop",
                elapsed_seconds=0.1,
                usage=TokenUsage(total_tokens=20),
            ),
            # Auto-verify failed on run 1, so agent loop gives model another turn to fix:
            ChatResponse(
                content="",
                model="fake/model",
                finish_reason="tool_calls",
                elapsed_seconds=0.1,
                tool_calls=[ToolCall(id="c2", name="run_tests", arguments="{}")],
                usage=TokenUsage(total_tokens=20),
            ),
            ChatResponse(
                content="All tests pass!\n<goal_status>COMPLETE</goal_status>",
                model="fake/model",
                finish_reason="stop",
                elapsed_seconds=0.1,
                usage=TokenUsage(total_tokens=20),
            ),
        ]
    )

    res = run_goal_loop(
        client=fake_client,  # type: ignore[arg-type]
        read_only_registry=ro_reg,
        write_registry=rw_reg,
        model="fake/model",
        objective=obj,
        max_iterations=steps,
        max_calls_per_iteration=6,
        workspace_root=tmp_path,
        tool_state=state,
        test_runner=runner,
    )
    assert res.completed is True
    assert res.iterations_run == 1
    assert "greet.py" in res.modified_files
    assert res.all_edits_verified is True
    assert (tmp_path / "greet.py").exists()
