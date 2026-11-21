"""Built-in workspace tools for file inspection, search, editing, and isolated testing."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from poseidon.context.checkpoints import WorkspaceCheckpointManager
from poseidon.indexing.semantic_index import SemanticCodeIndex
from poseidon.patching.unified_diff import (
    apply_unified_diff,
    extract_diff_payload,
    parse_unified_diff,
    sha256_text,
)
from poseidon.sandbox.docker_runner import TestRunner, TestRunResult
from poseidon.search.lexical import search_workspace_text
from poseidon.tools.registry import (
    ToolDefinition,
    ToolExecutionResult,
    ToolRegistry,
    ToolSessionState,
)
from poseidon.workspace import WorkspaceGuard


def _format_numbered_lines(content: str) -> str:
    """Add 1-based line numbers to file text to help models generate accurate unified diffs."""
    return "\n".join(f"{idx:4d} | {line}" for idx, line in enumerate(content.splitlines(), start=1))


def is_test_file_path(relative_path: str) -> bool:
    """Return True if a workspace path is a unit test file that should stay locked during repair."""
    path = Path(relative_path)
    name = path.name
    if name.startswith("test_") and name.endswith(".py"):
        return True
    if name.endswith("_test.py"):
        return True
    return "tests" in path.parts


def _format_new_file_as_diff(rel_path: str, content: str) -> str:
    """Render a new file's content as a unified diff preview for consistent user approval."""
    lines = content.splitlines()
    header = [
        "--- /dev/null",
        f"+++ b/{rel_path}",
        f"@@ -0,0 +1,{len(lines)} @@",
    ]
    return "\n".join([*header, *(f"+{line}" for line in lines)])


def register_core_tools(
    *,
    registry: ToolRegistry,
    workspace: WorkspaceGuard,
    state: ToolSessionState,
    confirm_patch: Callable[[str], bool] | None = None,
    test_runner: TestRunner | None = None,
    on_test_result: Callable[[TestRunResult], None] | None = None,
    semantic_index: SemanticCodeIndex | None = None,
    checkpoints: WorkspaceCheckpointManager | None = None,
    protect_tests: bool = True,
) -> None:
    """Register all built-in workspace tools strictly sandboxed to the workspace."""

    def handle_list_files(args: dict[str, Any]) -> ToolExecutionResult:
        subdir = args.get("subdir")
        subdir_str = str(subdir).strip() if subdir is not None else None
        files = workspace.list_files(subdir=subdir_str)
        if not files:
            return ToolExecutionResult(
                tool_name="list_files",
                ok=True,
                output="No files found in workspace.",
                summary="0 files",
            )
        return ToolExecutionResult(
            tool_name="list_files",
            ok=True,
            output="\n".join(files),
            summary=f"{len(files)} file(s)",
        )

    def handle_read_file(args: dict[str, Any]) -> ToolExecutionResult:
        rel_path = str(args["path"]).strip()
        content = workspace.read_text(rel_path)
        state.read_hashes[rel_path] = sha256_text(content)
        line_count = len(content.splitlines())
        numbered = _format_numbered_lines(content)
        return ToolExecutionResult(
            tool_name="read_file",
            ok=True,
            output=(
                f"File: {rel_path} ({line_count} lines)\n"
                "Note: line numbers on the left are for reference only — "
                "do NOT include them in diffs.\n"
                f"{numbered}"
            ),
            summary=f"read {rel_path} ({line_count} lines)",
        )

    def handle_read_lines(args: dict[str, Any]) -> ToolExecutionResult:
        rel_path = str(args["path"]).strip()
        start_line = int(args["start_line"])
        end_line = int(args["end_line"])
        sliced, total, full_text = workspace.read_lines(
            rel_path,
            start_line=start_line,
            end_line=end_line,
        )
        # Record full-file hash so apply_diff can safely verify version after range reads
        state.read_hashes[rel_path] = sha256_text(full_text)
        formatted = "\n".join(f"{line_no:4d} | {line}" for line_no, line in sliced)
        actual_end = sliced[-1][0] if sliced else start_line
        return ToolExecutionResult(
            tool_name="read_lines",
            ok=True,
            output=(
                f"File: {rel_path} (lines {start_line}-{actual_end} of {total})\n"
                "Note: line numbers on the left are for reference only — "
                "do NOT include them in diffs.\n"
                f"{formatted}"
            ),
            summary=f"read {rel_path} L{start_line}-{actual_end} ({total} total)",
        )

    def handle_search_text(args: dict[str, Any]) -> ToolExecutionResult:
        query = str(args["query"])
        regex = bool(args.get("regex", False))
        subdir = args.get("subdir")
        subdir_str = str(subdir).strip() if subdir is not None else None
        max_results = int(args.get("max_results", 50))
        res = search_workspace_text(
            workspace=workspace,
            query=query,
            regex=regex,
            max_results=max(1, min(max_results, 100)),
            subdir=subdir_str,
        )
        return ToolExecutionResult(
            tool_name="search_text",
            ok=True,
            output=res.format_for_model(),
            summary=(
                f"{len(res.hits)} hit(s) for {res.query!r} "
                f"({res.engine}, {res.elapsed_ms:.1f}ms)"
            ),
        )

    def handle_create_file(args: dict[str, Any]) -> ToolExecutionResult:
        rel_path = str(args["path"]).strip()
        content = str(args["content"])
        target = workspace.resolve_path(rel_path)

        if protect_tests and is_test_file_path(rel_path):
            return ToolExecutionResult(
                tool_name="create_file",
                ok=False,
                output=(
                    f"Error: '{rel_path}' is a protected unit-test path and cannot be created "
                    "during repair."
                ),
                summary=f"blocked creating protected test file {rel_path}",
            )

        if target.exists():
            return ToolExecutionResult(
                tool_name="create_file",
                ok=False,
                output=(
                    f"Error: '{rel_path}' already exists. Use read_file and apply_diff to edit "
                    "existing files."
                ),
                summary=f"refused overwrite of existing {rel_path}",
            )

        normalized = content if content.endswith("\n") else f"{content}\n"
        if confirm_patch is not None:
            preview = _format_new_file_as_diff(rel_path, normalized)
            if not confirm_patch(preview):
                return ToolExecutionResult(
                    tool_name="create_file",
                    ok=False,
                    output=f"User declined creating '{rel_path}'.",
                    summary=f"creation declined ({rel_path})",
                )

        if checkpoints is not None:
            checkpoints.capture(label=f"create {rel_path}", target_paths=[rel_path])
        workspace.write_text(rel_path, normalized)
        state.read_hashes[rel_path] = sha256_text(normalized)
        state.dirty_since_last_test = True
        if semantic_index is not None and semantic_index.cache_path.is_file():
            semantic_index.sync()
        line_count = len(normalized.splitlines())
        return ToolExecutionResult(
            tool_name="create_file",
            ok=True,
            output=f"Created new file '{rel_path}' ({line_count} lines).",
            summary=f"created {rel_path} ({line_count} lines)",
            modified_files=[rel_path],
        )

    def handle_apply_diff(args: dict[str, Any]) -> ToolExecutionResult:
        raw_diff = str(args["diff"])
        has_wrapper = "<diff>" in raw_diff or "```" in raw_diff
        diff_text = extract_diff_payload(raw_diff) if has_wrapper else raw_diff

        file_patches = parse_unified_diff(diff_text)
        for fp in file_patches:
            target = fp.target_path
            workspace.resolve_path(target)
            if protect_tests and is_test_file_path(target):
                return ToolExecutionResult(
                    tool_name="apply_diff",
                    ok=False,
                    output=(
                        f"Error: '{target}' is a protected unit-test file and cannot be modified "
                        "during repair. Fix the implementation code instead."
                    ),
                    summary=f"blocked edit to protected test file {target}",
                )
            if target not in state.read_hashes:
                return ToolExecutionResult(
                    tool_name="apply_diff",
                    ok=False,
                    output=(
                        f"Error: '{target}' has not been read yet. "
                        f"Call read_file or read_lines on '{target}' first so Poseidon can verify "
                        "the file version before patching."
                    ),
                    summary=f"rejected edit (must read {target} first)",
                )

        apply_unified_diff(
            workspace=workspace,
            diff_text=diff_text,
            expected_hashes=state.read_hashes,
            dry_run=True,
        )

        if confirm_patch is not None and not confirm_patch(diff_text):
            return ToolExecutionResult(
                tool_name="apply_diff",
                ok=False,
                output="User declined to apply this patch. Files were left unchanged.",
                summary="patch declined by user",
            )

        if checkpoints is not None:
            target_list = [fp.target_path for fp in file_patches]
            checkpoints.capture(label=f"patch {', '.join(target_list)}", target_paths=target_list)

        patch_res = apply_unified_diff(
            workspace=workspace,
            diff_text=diff_text,
            expected_hashes=state.read_hashes,
            dry_run=False,
        )

        for rel_path in patch_res.modified_files:
            state.read_hashes[rel_path] = sha256_text(workspace.read_text(rel_path))
        state.dirty_since_last_test = True
        if semantic_index is not None and semantic_index.cache_path.is_file():
            semantic_index.sync()

        files_csv = ", ".join(patch_res.modified_files)
        return ToolExecutionResult(
            tool_name="apply_diff",
            ok=True,
            output=(
                f"Applied {patch_res.hunks_applied} hunk(s) to {files_csv} "
                f"(+{patch_res.added_lines} / -{patch_res.removed_lines} lines)."
            ),
            summary=f"patched {files_csv} (+{patch_res.added_lines}/-{patch_res.removed_lines})",
            modified_files=patch_res.modified_files,
        )

    registry.register(
        ToolDefinition(
            name="list_files",
            description=(
                "List source files in the workspace (excluding secrets, .git, dependencies, "
                "and generated files)."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "subdir": {
                        "type": "string",
                        "description": "Optional relative subdirectory inside the workspace.",
                    }
                },
                "additionalProperties": False,
            },
            handler=handle_list_files,
        )
    )

    registry.register(
        ToolDefinition(
            name="read_file",
            description=(
                "Read the entire contents of a file inside the workspace with line numbers. "
                "Always read a file (via read_file or read_lines) before editing with apply_diff."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Relative path of the file inside the workspace.",
                    }
                },
                "required": ["path"],
                "additionalProperties": False,
            },
            handler=handle_read_file,
        )
    )

    registry.register(
        ToolDefinition(
            name="read_lines",
            description=(
                "Read a specific 1-based inclusive line range from a workspace file with line "
                "numbers. Prefer this over read_file when inspecting targeted sections."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Relative path of the file inside the workspace.",
                    },
                    "start_line": {
                        "type": "integer",
                        "minimum": 1,
                        "description": "First 1-based line number to read.",
                    },
                    "end_line": {
                        "type": "integer",
                        "minimum": 1,
                        "description": "Last 1-based line number to read (inclusive).",
                    },
                },
                "required": ["path", "start_line", "end_line"],
                "additionalProperties": False,
            },
            handler=handle_read_lines,
        )
    )

    registry.register(
        ToolDefinition(
            name="search_text",
            description=(
                "Search workspace source files for a text string or regex using ripgrep. "
                "Returns path, line number, and matching line for each hit. Automatically "
                "excludes secrets, Git data, dependencies, and generated files."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Literal text or regex pattern to search for.",
                    },
                    "regex": {
                        "type": "boolean",
                        "description": "True to treat query as regex (default false = literal).",
                    },
                    "subdir": {
                        "type": "string",
                        "description": "Optional relative subdirectory to narrow search.",
                    },
                    "max_results": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 100,
                        "description": "Maximum number of hits to return (default 50).",
                    },
                },
                "required": ["query"],
                "additionalProperties": False,
            },
            handler=handle_search_text,
        )
    )

    registry.register(
        ToolDefinition(
            name="create_file",
            description=(
                "Create a brand-new source file inside the workspace. Fails if the file already "
                "exists (use apply_diff for existing files)."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Relative path for the new file inside the workspace.",
                    },
                    "content": {
                        "type": "string",
                        "description": "Complete UTF-8 source text for the new file.",
                    },
                },
                "required": ["path", "content"],
                "additionalProperties": False,
            },
            handler=handle_create_file,
        )
    )

    registry.register(
        ToolDefinition(
            name="apply_diff",
            description=(
                "Apply a Git unified diff to an existing workspace file. "
                "Requires standard --- a/path, +++ b/path, and @@ hunk headers with exact context. "
                "The entire patch is validated atomically before writing. "
                "Unit test files are protected during repair."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "diff": {
                        "type": "string",
                        "description": (
                            "Unified diff string targeting files already read via "
                            "read_file or read_lines."
                        ),
                    }
                },
                "required": ["diff"],
                "additionalProperties": False,
            },
            handler=handle_apply_diff,
        )
    )

    if test_runner is not None:

        def handle_run_tests(_args: dict[str, Any]) -> ToolExecutionResult:
            if (
                state.last_test_result is not None
                and not state.dirty_since_last_test
                and state.last_test_result.verified
            ):
                prev = state.last_test_result
                return ToolExecutionResult(
                    tool_name="run_tests",
                    ok=True,
                    output=(
                        f"Cached result: no files changed since the last run_tests call, which "
                        f"already PASSED ({prev.passed}/{prev.total}). "
                        "Do not call run_tests again without editing files; summarize your work "
                        "and finish."
                    ),
                    summary=f"cached PASS ({prev.passed}/{prev.total}; no edits since last run)",
                )

            res = test_runner.run(workspace.root)
            state.last_test_result = res
            state.dirty_since_last_test = False
            if on_test_result is not None:
                on_test_result(res)
            summary = (
                f"PASS ({res.passed}/{res.total} passed)"
                if res.verified
                else f"UNVERIFIED ({res.unverified_reason or 'tests failed'})"
            )
            return ToolExecutionResult(
                tool_name="run_tests",
                ok=res.verified,
                output=res.format_for_model(),
                summary=summary,
            )

        registry.register(
            ToolDefinition(
                name="run_tests",
                description=(
                    "Run the full unit test suite inside a disposable network-isolated Docker "
                    "sandbox with no secrets. Returns exit status, pass/fail counts, failing "
                    "test names, and error messages."
                ),
                parameters={
                    "type": "object",
                    "properties": {},
                    "additionalProperties": False,
                },
                handler=handle_run_tests,
            )
        )

    if semantic_index is not None:

        def handle_semantic_search(args: dict[str, Any]) -> ToolExecutionResult:
            query = str(args["query"])
            top_k = int(args.get("top_k", 5))
            res = semantic_index.search(
                query,
                top_k=max(1, min(top_k, 10)),
                auto_sync=True,
            )
            return ToolExecutionResult(
                tool_name="semantic_search",
                ok=True,
                output=res.format_for_model(),
                summary=(
                    f"{len(res.hits)} match(es) for {res.query!r} "
                    f"({res.lookup_ms:.1f}ms)"
                ),
            )

        registry.register(
            ToolDefinition(
                name="semantic_search",
                description=(
                    "Find functions or code blocks by meaning using local code embeddings. "
                    "Returns file paths and exact line ranges so you can inspect them with "
                    "read_lines. Automatically refreshes the cached index when files change."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Plain-English description or symbol concept to find.",
                        },
                        "top_k": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": 10,
                            "description": "Number of top matching code blocks (default 5).",
                        },
                    },
                    "required": ["query"],
                    "additionalProperties": False,
                },
                handler=handle_semantic_search,
            )
        )
