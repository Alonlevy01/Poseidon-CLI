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


def draft_commit_message(
    workspace: WorkspaceGuard,
    client: object,
    model: str,
) -> tuple[bool, str]:
    """Read `git diff --staged` and draft a concise Conventional Commit message via LLM."""
    ok, diff_out = _run_git(
        workspace,
        ["diff", "--cached", "--", ".", ":(exclude).env*", ":(exclude)*.pem", ":(exclude)*.key"],
    )
    if not ok:
        return False, f"Could not inspect staged diff: {diff_out}"
    staged = diff_out.strip()
    if not staged:
        return False, "No staged changes found. Stage files with `git add` first."
    if len(staged) > MAX_DIFF_CHARS:
        staged = staged[:MAX_DIFF_CHARS] + "\n... [truncated]"

    resp = client.chat_completion(  # type: ignore[attr-defined]
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "Write ONE concise Conventional Commit subject line (<= 72 chars, e.g. "
                    "`feat(cli): add dry-run diff preview`) describing the staged diff. "
                    "Return ONLY the commit message line, no quotes or markdown."
                ),
            },
            {"role": "user", "content": staged},
        ],
        temperature=0.1,
    )
    msg = resp.content.strip().splitlines()[0].strip("`\"' ")
    if not msg:
        return False, "Model returned an empty commit message."
    return True, msg


def run_git_commit(workspace: WorkspaceGuard, message: str) -> tuple[bool, str]:
    """Commit staged changes using `git commit -m <message>` with shell=False."""
    cleaned = message.strip()
    if not cleaned:
        return False, "Commit message cannot be empty."
    return _run_git(workspace, ["commit", "-m", cleaned])
