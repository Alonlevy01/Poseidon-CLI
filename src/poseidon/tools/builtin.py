"""Built-in Milestone 3 workspace tools: read_file and apply_diff."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from poseidon.patching.unified_diff import (
    apply_unified_diff,
    extract_diff_payload,
    parse_unified_diff,
    sha256_text,
)
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


def register_core_tools(
    *,
    registry: ToolRegistry,
    workspace: WorkspaceGuard,
    state: ToolSessionState,
    confirm_patch: Callable[[str], bool] | None = None,
) -> None:
    """Register read_file and apply_diff strictly sandboxed to the workspace."""

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

    def handle_apply_diff(args: dict[str, Any]) -> ToolExecutionResult:
        raw_diff = str(args["diff"])
        has_wrapper = "<diff>" in raw_diff or "```" in raw_diff
        diff_text = extract_diff_payload(raw_diff) if has_wrapper else raw_diff

        # Parse first to check workspace containment and enforce read-before-edit
        file_patches = parse_unified_diff(diff_text)
        for fp in file_patches:
            target = fp.target_path
            workspace.resolve_path(target)
            if target not in state.read_hashes:
                return ToolExecutionResult(
                    tool_name="apply_diff",
                    ok=False,
                    output=(
                        f"Error: '{target}' has not been read yet. "
                        f"Call read_file(path='{target}') first so Poseidon can verify "
                        "the file version before patching."
                    ),
                    summary=f"rejected edit (must read {target} first)",
                )

        # Dry-run validate stale hashes + exact hunks before asking user approval
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

        # Apply atomically (re-verifying hashes in case file changed during prompt)
        patch_res = apply_unified_diff(
            workspace=workspace,
            diff_text=diff_text,
            expected_hashes=state.read_hashes,
            dry_run=False,
        )

        # Refresh recorded hashes so subsequent diffs in the same run can apply cleanly
        for rel_path in patch_res.modified_files:
            state.read_hashes[rel_path] = sha256_text(workspace.read_text(rel_path))

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
            name="read_file",
            description=(
                "Read the contents of a file inside the workspace with line numbers. "
                "Always read a file before editing it with apply_diff."
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
            name="apply_diff",
            description=(
                "Apply a Git unified diff to an existing workspace file. "
                "Requires standard --- a/path, +++ b/path, and @@ hunk headers with exact context. "
                "The entire patch is validated atomically before writing."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "diff": {
                        "type": "string",
                        "description": (
                            "Unified diff string targeting files already read via read_file."
                        ),
                    }
                },
                "required": ["diff"],
                "additionalProperties": False,
            },
            handler=handle_apply_diff,
        )
    )
