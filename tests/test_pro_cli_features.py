"""Unit tests for Pro-CLI features: doctor, @file mentions, Global/Workspace /cmds, and /cost."""

from __future__ import annotations

from pathlib import Path

import pytest
from starlette.testclient import TestClient

from poseidon.config import Settings
from poseidon.context.custom_commands import CustomCommandStore
from poseidon.context.mentions import expand_file_mentions
from poseidon.tools.registry import ToolSessionState
from poseidon.web.server import create_web_app
from poseidon.workspace import WorkspaceGuard


def test_file_mentions_expand_safe_files_and_block_secrets(tmp_path: Path) -> None:
    (tmp_path / "analytics.py").write_text("def total(xs):\n    return sum(xs)\n", encoding="utf-8")
    (tmp_path / ".env").write_text("OPENROUTER_API_KEY=secret\n", encoding="utf-8")

    guard = WorkspaceGuard(tmp_path)
    state = ToolSessionState()
    res = expand_file_mentions(
        "Refactor @analytics.py and ignore @.env and @../outside.py",
        guard,
        state,
    )
    assert res.loaded_files == ["analytics.py"]
    assert "def total(xs):" in res.expanded_prompt
    assert "analytics.py" in state.read_hashes
    assert "OPENROUTER_API_KEY" not in res.expanded_prompt
    assert len(res.skipped) == 2


def test_custom_commands_support_global_and_workspace_scopes(tmp_path: Path) -> None:
    ws_a = tmp_path / "project_a"
    ws_b = tmp_path / "project_b"
    ws_a.mkdir()
    ws_b.mkdir()
    global_dir = tmp_path / "global_cmds"

    store_a = CustomCommandStore(ws_a, global_dir=global_dir)
    store_a.save(
        name="review",
        scope="global",
        description="Code review",
        template="Review code for bugs. Focus: $ARGUMENTS",
    )

    # A completely different project sees the exact same Global `/review` command!
    store_b = CustomCommandStore(ws_b, global_dir=global_dir)
    rendered = store_b.render("/review authentication flow")
    assert rendered is not None
    cmd, expanded = rendered
    assert cmd.scope == "global"
    assert expanded == "Review code for bugs. Focus: authentication flow"

    # Workspace command overrides global command of the same name only in project_b
    store_b.save(
        name="review",
        scope="workspace",
        description="Project B review",
        template="Strict Project B review: $ARGUMENTS",
    )
    rendered_b = store_b.render("/review api")
    assert rendered_b is not None
    assert rendered_b[0].scope == "workspace"
    assert "Project B" in rendered_b[1]

    # Deleting workspace override reveals the global command again
    assert store_b.delete(name="review", scope="workspace") is True
    assert store_b.get("review") is not None
    assert store_b.get("review").scope == "global"  # type: ignore[union-attr]


def test_web_api_doctor_cost_and_custom_commands(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("POSEIDON_GLOBAL_COMMANDS_DIR", str(tmp_path / "global_cmds"))
    app = create_web_app(workspace=tmp_path, settings=Settings())
    client = TestClient(app)

    # Save a global custom command via REST API
    saved = client.post(
        "/api/custom-commands/save",
        json={
            "name": "sec",
            "scope": "global",
            "description": "Security audit",
            "template": "Audit for security issues: $ARGUMENTS",
        },
    )
    assert saved.status_code == 200
    assert any(c["name"] == "/sec" for c in saved.json()["state"]["slash_commands"])

    # /cmd list, /cost, and /doctor all work via /api/command
    cmd_list = client.post("/api/command", json={"command": "/cmd list"}).json()
    assert cmd_list["ok"] is True
    assert any(r[0] == "/sec" and r[1] == "global" for r in cmd_list["rows"])

    cost_res = client.post("/api/command", json={"command": "/cost"}).json()
    assert cost_res["ok"] is True
    assert any(r[0] == "Total Cost (USD)" for r in cost_res["rows"])

    doc_res = client.post("/api/command", json={"command": "/doctor"}).json()
    assert doc_res["kind"] == "table"
    assert len(doc_res["rows"]) >= 6

    # Delete the custom command via REST API
    deleted = client.post("/api/custom-commands/delete", json={"name": "sec", "scope": "global"})
    assert deleted.status_code == 200
    assert not any(c["name"] == "/sec" for c in deleted.json()["state"]["slash_commands"])


def test_vision_image_mentions_and_anthropic_translation(tmp_path: Path) -> None:
    from poseidon.llm.openrouter import _convert_messages_to_anthropic

    # Tiny valid 1x1 PNG header bytes
    png_bytes = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR"
    (tmp_path / "shot.png").write_bytes(png_bytes)

    guard = WorkspaceGuard(tmp_path)
    res = expand_file_mentions("Check @shot.png for layout bugs", guard)
    assert len(res.images) == 1
    assert res.images[0]["mime_type"] == "image/png"
    assert res.images[0]["data_url"].startswith("data:image/png;base64,")

    _, anth_msgs = _convert_messages_to_anthropic(
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Inspect this"},
                    {"type": "image_url", "image_url": {"url": res.images[0]["data_url"]}},
                ],
            }
        ]
    )
    assert anth_msgs[0]["content"][0] == {"type": "text", "text": "Inspect this"}
    assert anth_msgs[0]["content"][1]["type"] == "image"
    assert anth_msgs[0]["content"][1]["source"]["media_type"] == "image/png"

