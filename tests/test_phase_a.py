"""Unit tests for Phase A: project memory, JSONL session persistence, and read-only Plan Mode."""

from pathlib import Path

from poseidon.context.memory import load_project_memory, starter_poseidon_md
from poseidon.context.sessions import SessionStore
from poseidon.core.agent import build_system_prompt
from poseidon.tools.builtin import register_core_tools
from poseidon.tools.registry import ToolRegistry, ToolSessionState
from poseidon.workspace import WorkspaceGuard


def test_project_memory_loads_poseidon_and_agents_md(tmp_path: Path) -> None:
    empty = load_project_memory(tmp_path)
    assert empty.sources == []
    assert empty.content == ""

    (tmp_path / "POSEIDON.md").write_text(starter_poseidon_md("Demo"), encoding="utf-8")
    (tmp_path / "AGENTS.md").write_text("Always use type hints.", encoding="utf-8")

    loaded = load_project_memory(tmp_path)
    assert loaded.sources == ["POSEIDON.md", "AGENTS.md"]
    assert "POSEIDON.md" in loaded.content
    assert "Always use type hints." in loaded.content

    prompt = build_system_prompt(project_memory=loaded.content, plan_mode=True)
    assert "Project Memory" in prompt
    assert "PLAN MODE ACTIVE" in prompt


def test_jsonl_session_save_list_and_resume(tmp_path: Path) -> None:
    store = SessionStore(tmp_path)
    msgs = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "Refactor the analytics module"},
        {"role": "assistant", "content": "Sure, inspecting files now."},
    ]

    summary = store.save(
        session_id="my refactor session",
        model="nvidia/nemotron-3.5-super-vl:free",
        messages=msgs,
        plan_mode=True,
    )
    assert summary.session_id == "my-refactor-session"
    assert summary.message_count == 2
    assert summary.plan_mode is True

    listed = store.list_sessions()
    assert len(listed) == 1
    assert store.latest_session_id() == "my-refactor-session"

    loaded = store.load("my-refactor-session")
    assert loaded.summary.model == "nvidia/nemotron-3.5-super-vl:free"
    assert loaded.summary.plan_mode is True
    assert len(loaded.messages) == 3
    assert loaded.messages[1]["content"] == "Refactor the analytics module"


def test_plan_mode_physically_omits_write_tools(tmp_path: Path) -> None:
    guard = WorkspaceGuard(tmp_path)
    registry = ToolRegistry()
    register_core_tools(
        registry=registry,
        workspace=guard,
        state=ToolSessionState(),
        read_only=True,
    )
    names = registry.names()
    assert "read_file" in names
    assert "read_lines" in names
    assert "search_text" in names
    assert "list_files" in names
    assert "apply_diff" not in names
    assert "create_file" not in names
