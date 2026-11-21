"""Unit tests for signature feature: deterministic observation masking & /undo checkpoints."""

from pathlib import Path

from poseidon.context.checkpoints import WorkspaceCheckpointManager
from poseidon.context.compactor import compact_messages
from poseidon.workspace import WorkspaceGuard


def test_observation_masking_compacts_old_tool_outputs_and_preserves_recent() -> None:
    bulky = "line of source code\n" * 60
    messages = [
        {"role": "system", "content": "System prompt"},
        {"role": "user", "content": "Fix the bug"},
        {"role": "assistant", "content": None, "tool_calls": [{"id": "1"}]},
        {"role": "tool", "tool_call_id": "1", "name": "read_file", "content": bulky},
        {"role": "assistant", "content": None, "tool_calls": [{"id": "2"}]},
        {"role": "tool", "tool_call_id": "2", "name": "search_text", "content": bulky},
        {"role": "assistant", "content": None, "tool_calls": [{"id": "3"}]},
        {"role": "tool", "tool_call_id": "3", "name": "read_lines", "content": bulky},
    ]

    compacted, stats = compact_messages(
        messages,
        keep_recent_tool_outputs=1,
        min_chars_to_mask=200,
        in_place=False,
    )

    assert stats.masked_count == 2
    assert stats.chars_saved > 1500
    assert stats.estimated_tokens_saved > 350
    assert compacted[0]["content"] == "System prompt"
    assert compacted[1]["content"] == "Fix the bug"
    assert compacted[3]["content"].startswith("[masked previous read_file output:")
    assert compacted[5]["content"].startswith("[masked previous search_text output:")
    # Most recent tool output must stay completely verbatim
    assert compacted[7]["content"] == bulky


def test_workspace_checkpoints_undo_edits_and_deletes_created_files(tmp_path: Path) -> None:
    guard = WorkspaceGuard(tmp_path)
    guard.write_text("app.py", "x = 1\n")
    mgr = WorkspaceCheckpointManager(guard)

    # Checkpoint 1: edit existing file
    mgr.capture(label="patch app.py", target_paths=["app.py"])
    guard.write_text("app.py", "x = 2\n")

    # Checkpoint 2: create brand-new file
    mgr.capture(label="create helper.py", target_paths=["helper.py"])
    guard.write_text("helper.py", "y = 99\n")
    assert len(mgr.list_checkpoints()) == 2

    # Undo #2 -> deletes helper.py
    cp2, actions2 = mgr.undo()
    assert cp2 is not None and cp2.id == 2
    assert actions2 == ["deleted helper.py"]
    assert not (tmp_path / "helper.py").exists()

    # Undo #1 -> restores app.py back to x = 1
    cp1, actions1 = mgr.undo()
    assert cp1 is not None and cp1.id == 1
    assert actions1 == ["restored app.py"]
    assert guard.read_text("app.py") == "x = 1\n"

    # Undo on empty stack -> safe no-op
    cp0, actions0 = mgr.undo()
    assert cp0 is None and actions0 == []
