"""Read-only Git repository inspection tools (`git_status` and `git_diff`) for Poseidon."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from poseidon.tools.registry import ToolDefinition, ToolExecutionResult, ToolRegistry
from poseidon.workspace import WorkspaceGuard

MAX_DIFF_CHARS = 8000


def _run_git(workspace: WorkspaceGuard, args: list[str]) -> tuple[bool, str]:
    if shutil.which("git") is None:
        return False, "git binary is not installed in this container."
    proc = subprocess.run(
        ["git", *args],
        cwd=str(workspace.root),
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout).strip()
        return False, err or f"git exited with code {proc.returncode}"
    return True, proc.stdout


def register_git_tools(registry: ToolRegistry, workspace: WorkspaceGuard) -> None:
    """Register strictly read-only `git_status` and `git_diff` tools."""

    def handle_git_status(_args: dict[str, object]) -> ToolExecutionResult:
        ok, out = _run_git(workspace, ["status", "--short", "--branch"])
        if not ok:
            return ToolExecutionResult(
                tool_name="git_status",
                ok=False,
                output=f"git_status unavailable: {out}",
                summary="not a git repo",
            )
        lines = [
            line
            for line in out.splitlines()
            if not workspace.is_secret_path(Path(line[3:].strip()))
        ]
        cleaned = "\n".join(lines).strip() or "(clean working tree)"
        return ToolExecutionResult(
            tool_name="git_status",
            ok=True,
            output=cleaned,
            summary=f"{max(0, len(lines) - 1)} changed path(s)",
        )

    def handle_git_diff(args: dict[str, object]) -> ToolExecutionResult:
        path_arg = str(args.get("path") or "").strip()
        staged = bool(args.get("staged", False))
        cmd = ["diff", "--cached"] if staged else ["diff"]
        if path_arg:
            resolved = workspace.resolve_path(path_arg)
            rel = resolved.relative_to(workspace.root).as_posix()
            cmd.extend(["--", rel])
        else:
            cmd.extend(["--", ".", ":(exclude).env*", ":(exclude)*.pem", ":(exclude)*.key"])

        ok, out = _run_git(workspace, cmd)
        if not ok:
            return ToolExecutionResult(
                tool_name="git_diff",
                ok=False,
                output=f"git_diff unavailable: {out}",
                summary="git_diff error",
            )
        text = out.strip() or "(no diff)"
        if len(text) > MAX_DIFF_CHARS:
            text = text[:MAX_DIFF_CHARS] + f"\n\n... [diff truncated at {MAX_DIFF_CHARS} chars]"
        return ToolExecutionResult(
            tool_name="git_diff",
            ok=True,
            output=text,
            summary=f"{len(text)} chars diff",
        )

    registry.register(
        ToolDefinition(
            name="git_status",
            description="Inspect Git branch and modified/untracked workspace files (read-only).",
            parameters={"type": "object", "properties": {}},
            handler=handle_git_status,
        )
    )
    registry.register(
        ToolDefinition(
            name="git_diff",
            description="Inspect unstaged or staged Git diff for the workspace or a single file.",
            parameters={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Optional workspace-relative file path.",
                    },
                    "staged": {
                        "type": "boolean",
                        "description": "If true, inspect staged (`--cached`) diff instead.",
                    },
                },
            },
            handler=handle_git_diff,
        )
    )
