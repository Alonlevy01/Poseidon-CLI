"""Unit tests for ToolRegistry and Milestone 3 built-in tools (read_file, apply_diff)."""

import json
from pathlib import Path

from poseidon.tools.builtin import register_core_tools
from poseidon.tools.registry import ToolRegistry, ToolSessionState
from poseidon.workspace import WorkspaceGuard

SAMPLE_PY = """def multiply(a: int, b: int) -> int:
    return a + b
"""


def _build_registry(tmp_path: Path, *, confirm: bool = True) -> tuple[WorkspaceGuard, ToolRegistry]:
    guard = WorkspaceGuard(tmp_path)
    guard.write_text("math_ops.py", SAMPLE_PY)
    registry = ToolRegistry()
    state = ToolSessionState()
    register_core_tools(
        registry=registry,
        workspace=guard,
        state=state,
        confirm_patch=lambda _diff: confirm,
    )
    return guard, registry


def test_registry_exports_openrouter_schemas(tmp_path: Path) -> None:
    _, registry = _build_registry(tmp_path)
    schemas = registry.schemas()
    names = [s["function"]["name"] for s in schemas]
    assert names == ["read_file", "apply_diff"]
    assert all(s["type"] == "function" for s in schemas)


def test_read_then_apply_diff_succeeds(tmp_path: Path) -> None:
    guard, registry = _build_registry(tmp_path)

    read_res = registry.execute("read_file", json.dumps({"path": "math_ops.py"}))
    assert read_res.ok
    assert "return a + b" in read_res.output

    diff = (
        "--- a/math_ops.py\n"
        "+++ b/math_ops.py\n"
        "@@ -1,2 +1,2 @@\n"
        " def multiply(a: int, b: int) -> int:\n"
        "-    return a + b\n"
        "+    return a * b\n"
    )
    patch_res = registry.execute("apply_diff", json.dumps({"diff": diff}))
    assert patch_res.ok
    assert patch_res.modified_files == ["math_ops.py"]
    assert "return a * b" in guard.read_text("math_ops.py")


def test_apply_diff_requires_read_file_first(tmp_path: Path) -> None:
    guard, registry = _build_registry(tmp_path)
    diff = (
        "--- a/math_ops.py\n"
        "+++ b/math_ops.py\n"
        "@@ -1,2 +1,2 @@\n"
        " def multiply(a: int, b: int) -> int:\n"
        "-    return a + b\n"
        "+    return a * b\n"
    )
    res = registry.execute("apply_diff", json.dumps({"diff": diff}))
    assert not res.ok
    assert "has not been read yet" in res.output
    assert guard.read_text("math_ops.py") == SAMPLE_PY


def test_apply_diff_rejects_stale_file_after_external_change(tmp_path: Path) -> None:
    guard, registry = _build_registry(tmp_path)
    assert registry.execute("read_file", json.dumps({"path": "math_ops.py"})).ok

    # External edit happens after model read the file
    changed = SAMPLE_PY + "\n# external edit\n"
    guard.write_text("math_ops.py", changed)

    diff = (
        "--- a/math_ops.py\n"
        "+++ b/math_ops.py\n"
        "@@ -1,2 +1,2 @@\n"
        " def multiply(a: int, b: int) -> int:\n"
        "-    return a + b\n"
        "+    return a * b\n"
    )
    res = registry.execute("apply_diff", json.dumps({"diff": diff}))
    assert not res.ok
    assert "Stale file check failed" in res.output
    assert guard.read_text("math_ops.py") == changed


def test_tools_block_workspace_escape_and_bad_json(tmp_path: Path) -> None:
    _, registry = _build_registry(tmp_path)

    esc = registry.execute("read_file", json.dumps({"path": "../secret.env"}))
    assert not esc.ok
    assert "escapes workspace" in esc.output

    bad_json = registry.execute("read_file", "{not valid json")
    assert not bad_json.ok
    assert "invalid JSON" in bad_json.output

    unknown = registry.execute("delete_everything", "{}")
    assert not unknown.ok
    assert "unknown tool" in unknown.output
