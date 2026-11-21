"""Unit tests for Milestone 5 lexical search (ripgrep), exclusions, and range reading."""

from pathlib import Path

import pytest

from poseidon.search.lexical import search_workspace_text
from poseidon.workspace import WorkspaceGuard, WorkspaceSecurityError


def test_search_text_finds_matches_and_excludes_secrets_git_deps_generated(tmp_path: Path) -> None:
    guard = WorkspaceGuard(tmp_path)
    guard.write_text("parser.py", "def parse_expenses(raw):\n    # FIND_ME_TOKEN\n    return {}\n")
    guard.write_text("reports.py", "def format_report():\n    return 'FIND_ME_TOKEN'\n")

    # Files that MUST be excluded from searches and list_files
    (tmp_path / ".env").write_text("API_KEY=FIND_ME_TOKEN\n", encoding="utf-8")
    (tmp_path / "private.key").write_text("FIND_ME_TOKEN\n", encoding="utf-8")
    (tmp_path / "app_secrets.txt").write_text("FIND_ME_TOKEN\n", encoding="utf-8")
    git_dir = tmp_path / ".git"
    git_dir.mkdir()
    (git_dir / "config").write_text("FIND_ME_TOKEN\n", encoding="utf-8")
    nm_dir = tmp_path / "node_modules" / "pkg"
    nm_dir.mkdir(parents=True)
    (nm_dir / "index.js").write_text("FIND_ME_TOKEN\n", encoding="utf-8")
    dist_dir = tmp_path / "dist"
    dist_dir.mkdir()
    (dist_dir / "bundle.min.js").write_text("FIND_ME_TOKEN\n", encoding="utf-8")

    assert guard.list_files() == ["parser.py", "reports.py"]

    res = search_workspace_text(workspace=guard, query="FIND_ME_TOKEN")
    hit_paths = sorted({h.path for h in res.hits})
    assert hit_paths == ["parser.py", "reports.py"]
    assert any(h.path == "parser.py" and h.line_number == 2 for h in res.hits)
    assert any(h.path == "reports.py" and h.line_number == 2 for h in res.hits)


def test_workspace_blocks_reading_secret_files_directly(tmp_path: Path) -> None:
    guard = WorkspaceGuard(tmp_path)
    (tmp_path / ".env").write_text("OPENROUTER_API_KEY=abc\n", encoding="utf-8")
    (tmp_path / "tls.pem").write_text("PRIVATE\n", encoding="utf-8")

    with pytest.raises(WorkspaceSecurityError, match="secret/credential"):
        guard.read_text(".env")
    with pytest.raises(WorkspaceSecurityError, match="secret/credential"):
        guard.read_text("tls.pem")


def test_workspace_read_lines_slice(tmp_path: Path) -> None:
    guard = WorkspaceGuard(tmp_path)
    guard.write_text("sample.py", "line1\nline2\nline3\nline4\nline5\n")

    sliced, total, _ = guard.read_lines("sample.py", start_line=2, end_line=4)
    assert total == 5
    assert sliced == [(2, "line2"), (3, "line3"), (4, "line4")]
