"""Unit tests for Phase C: Python AST syntax blocking & ruff F821 undefined-name preflight."""

import json
from pathlib import Path

from poseidon.patching.python_preflight import check_python_source
from poseidon.tools.builtin import register_core_tools
from poseidon.tools.registry import ToolRegistry, ToolSessionState
from poseidon.workspace import WorkspaceGuard


def test_preflight_blocks_syntax_error_before_writing_disk(tmp_path: Path) -> None:
    guard = WorkspaceGuard(tmp_path)
    original = "def add(a: int, b: int) -> int:\n    return a + b\n"
    guard.write_text("calc.py", original)

    registry = ToolRegistry()
    register_core_tools(
        registry=registry,
        workspace=guard,
        state=ToolSessionState(),
        enable_preflight=True,
    )
    assert registry.execute("read_file", json.dumps({"path": "calc.py"})).ok

    broken_diff = (
        "--- a/calc.py\n"
        "+++ b/calc.py\n"
        "@@ -1,2 +1,2 @@\n"
        " def add(a: int, b: int) -> int:\n"
        "-    return a + b\n"
        "+    return a +\n"
    )
    res = registry.execute("apply_diff", json.dumps({"diff": broken_diff}))
    assert not res.ok
    assert "SyntaxError" in res.output
    # Disk file must stay completely untouched!
    assert guard.read_text("calc.py") == original


def test_preflight_surfaces_undefined_name_warning_on_missing_import(tmp_path: Path) -> None:
    guard = WorkspaceGuard(tmp_path)
    guard.write_text("entry.py", "def run(x):\n    return x\n")

    registry = ToolRegistry()
    register_core_tools(
        registry=registry,
        workspace=guard,
        state=ToolSessionState(),
        enable_preflight=True,
    )
    assert registry.execute("read_file", json.dumps({"path": "entry.py"})).ok

    missing_import_diff = (
        "--- a/entry.py\n"
        "+++ b/entry.py\n"
        "@@ -1,2 +1,2 @@\n"
        " def run(x):\n"
        "-    return x\n"
        "+    return missing_helper_fn(x)\n"
    )
    res = registry.execute("apply_diff", json.dumps({"diff": missing_import_diff}))
    assert res.ok
    assert "F821" in res.output
    assert "missing_helper_fn" in res.output


def test_preflight_passes_clean_python_without_warnings() -> None:
    pf = check_python_source("ok.py", "def square(n: int) -> int:\n    return n * n\n")
    assert pf.ok
    assert pf.warnings == []
