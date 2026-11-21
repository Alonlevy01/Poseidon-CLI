"""Unit tests for WorkspaceGuard path containment."""

from pathlib import Path

import pytest

from poseidon.workspace import WorkspaceGuard, WorkspaceSecurityError


def test_read_write_and_list_inside_workspace(tmp_path: Path) -> None:
    guard = WorkspaceGuard(tmp_path)
    written = guard.write_text("ledger.py", "print('ok')\n")

    assert written == (tmp_path / "ledger.py").resolve()
    assert guard.read_text("ledger.py") == "print('ok')\n"
    assert guard.list_files() == ["ledger.py"]


def test_blocks_parent_directory_traversal(tmp_path: Path) -> None:
    guard = WorkspaceGuard(tmp_path)
    with pytest.raises(WorkspaceSecurityError, match="escapes workspace"):
        guard.resolve_path("../outside.py")


def test_blocks_absolute_path(tmp_path: Path) -> None:
    guard = WorkspaceGuard(tmp_path)
    with pytest.raises(WorkspaceSecurityError, match="Absolute paths"):
        guard.resolve_path("/etc/passwd")


def test_blocks_workspace_root_as_file(tmp_path: Path) -> None:
    guard = WorkspaceGuard(tmp_path)
    with pytest.raises(WorkspaceSecurityError, match="workspace root"):
        guard.resolve_path(".")


def test_list_files_excludes_hidden_files(tmp_path: Path) -> None:
    guard = WorkspaceGuard(tmp_path)
    (tmp_path / ".env").write_text("SECRET=1", encoding="utf-8")
    (tmp_path / ".gitkeep").write_text("", encoding="utf-8")
    guard.write_text("app.py", "x = 1\n")

    assert guard.list_files() == ["app.py"]
