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
    assert names == [
        "list_files",
        "read_file",
        "read_lines",
        "search_text",
        "create_file",
        "apply_diff",
        "git_status",
        "git_diff",
        "web_search",
        "fetch_url",
    ]
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


def test_apply_diff_protects_unit_test_files_during_repair(tmp_path: Path) -> None:
    guard, registry = _build_registry(tmp_path)
    guard.write_text("test_math_ops.py", "def test_mul():\n    assert False\n")
    assert registry.execute("read_file", json.dumps({"path": "test_math_ops.py"})).ok

    diff = (
        "--- a/test_math_ops.py\n"
        "+++ b/test_math_ops.py\n"
        "@@ -1,2 +1,2 @@\n"
        " def test_mul():\n"
        "-    assert False\n"
        "+    assert True\n"
    )
    res = registry.execute("apply_diff", json.dumps({"diff": diff}))
    assert not res.ok
    assert "protected unit-test file" in res.output
    assert "assert False" in guard.read_text("test_math_ops.py")


def test_list_files_read_lines_search_text_and_create_file(tmp_path: Path) -> None:
    guard, registry = _build_registry(tmp_path)

    listed = registry.execute("list_files", "{}")
    assert listed.ok
    assert "math_ops.py" in listed.output

    created = registry.execute(
        "create_file",
        json.dumps({"path": "formatter.py", "content": "def fmt(n):\n    return f'${n:.2f}'\n"}),
    )
    assert created.ok
    assert created.modified_files == ["formatter.py"]

    # Overwriting existing file via create_file must be refused
    dup = registry.execute(
        "create_file",
        json.dumps({"path": "formatter.py", "content": "x = 1\n"}),
    )
    assert not dup.ok
    assert "already exists" in dup.output

    # Search finds symbol in new file
    found = registry.execute("search_text", json.dumps({"query": "def fmt"}))
    assert found.ok
    assert "formatter.py:1:" in found.output

    # read_lines works and records hash so apply_diff succeeds immediately afterward
    ranged = registry.execute(
        "read_lines",
        json.dumps({"path": "formatter.py", "start_line": 1, "end_line": 2}),
    )
    assert ranged.ok
    assert "def fmt" in ranged.output
    diff = (
        "--- a/formatter.py\n"
        "+++ b/formatter.py\n"
        "@@ -1,2 +1,2 @@\n"
        " def fmt(n):\n"
        "-    return f'${n:.2f}'\n"
        "+    return f'USD {n:.2f}'\n"
    )
    patched = registry.execute("apply_diff", json.dumps({"diff": diff}))
    assert patched.ok
    assert "USD" in guard.read_text("formatter.py")


def test_run_tests_short_circuits_redundant_calls_without_edits(tmp_path: Path) -> None:
    from poseidon.sandbox.docker_runner import TestRunResult

    class CountingRunner:
        def __init__(self) -> None:
            self.calls = 0

        @property
        def timeout_seconds(self) -> int:
            return 15

        def run(
            self,
            workspace_root: Path,
            *,
            test_args: list[str] | None = None,
            extra_files: dict[str, str] | None = None,
        ) -> TestRunResult:
            self.calls += 1
            return TestRunResult(
                command="pytest -v --tb=short",
                timeout_seconds=15,
                exit_code=0,
                passed=3,
                total=3,
                verified=True,
            )

    guard = WorkspaceGuard(tmp_path)
    runner = CountingRunner()
    registry = ToolRegistry()
    register_core_tools(
        registry=registry,
        workspace=guard,
        state=ToolSessionState(),
        test_runner=runner,
    )

    first = registry.execute("run_tests", "{}")
    second = registry.execute("run_tests", "{}")
    assert first.ok and second.ok
    assert runner.calls == 1
    assert "no files changed" in second.output.lower()
