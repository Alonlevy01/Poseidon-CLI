"""Safe `@file` mention expansion for both CLI and Web UI prompts.

Detects workspace file mentions like `@analytics.py` or `@src/app.py`, validates every path
through `WorkspaceGuard` (blocking secrets and path traversal), appends numbered contents,
and records SHA-256 read hashes so the agent can patch mentioned files without wasting a
separate `read_file` tool call.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from poseidon.patching.unified_diff import sha256_text
from poseidon.tools.registry import ToolSessionState
from poseidon.workspace import WorkspaceGuard, WorkspaceSecurityError

_MENTION_RE = re.compile(r"(?<![\w.-])@(\.?[A-Za-z0-9_./\\-]+\.[A-Za-z0-9_]+|\.[A-Za-z0-9_.-]+)")
MAX_MENTION_CHARS_PER_FILE = 6000
MAX_MENTION_FILES = 5


@dataclass
class MentionExpansion:
    """Result of expanding `@file` references in a user prompt."""

    expanded_prompt: str
    loaded_files: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)


def expand_file_mentions(
    prompt: str,
    workspace: WorkspaceGuard,
    tool_state: ToolSessionState | None = None,
) -> MentionExpansion:
    """Expand valid workspace `@file` mentions into grounded numbered context blocks."""
    raw_matches = _MENTION_RE.findall(prompt)
    if not raw_matches:
        return MentionExpansion(expanded_prompt=prompt)

    seen: set[str] = set()
    loaded: list[str] = []
    skipped: list[str] = []
    blocks: list[str] = []

    for raw in raw_matches:
        rel = raw.replace("\\", "/")
        if rel.startswith("./"):
            rel = rel[2:]
        if rel in seen:
            continue
        seen.add(rel)
        if len(loaded) >= MAX_MENTION_FILES:
            skipped.append(f"{rel} (limit {MAX_MENTION_FILES} files)")
            continue
        try:
            content = workspace.read_text(rel)
        except (WorkspaceSecurityError, FileNotFoundError, OSError) as exc:
            skipped.append(f"{rel} ({exc})")
            continue

        if tool_state is not None:
            tool_state.read_hashes[rel] = sha256_text(content)

        lines = content.splitlines()
        numbered = "\n".join(f"{i:>4} | {line}" for i, line in enumerate(lines, start=1))
        if len(numbered) > MAX_MENTION_CHARS_PER_FILE:
            numbered = (
                numbered[:MAX_MENTION_CHARS_PER_FILE]
                + f"\n... [truncated at {MAX_MENTION_CHARS_PER_FILE} chars]"
            )
        blocks.append(f"### Referenced File: `{rel}` ({len(lines)} lines)\n```\n{numbered}\n```")
        loaded.append(rel)

    if not blocks:
        return MentionExpansion(expanded_prompt=prompt, loaded_files=[], skipped=skipped)

    suffix = "\n\n---\nAttached `@file` context (already read):\n\n" + "\n\n".join(blocks)
    return MentionExpansion(
        expanded_prompt=prompt + suffix,
        loaded_files=loaded,
        skipped=skipped,
    )
