"""Deterministic observation masking for coding-agent context window compaction.

Based on empirical coding-agent context engineering research (e.g., JetBrains / SWE-bench),
environment observations (file reads, search dumps, test logs) make up ~80% of prompt tokens.
Masking older verbose tool outputs while preserving system instructions, reasoning/tool-call
history, and the latest K observations cuts token usage without lossy LLM summarization.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class CompactionStats(BaseModel):
    """Metrics describing how much context was compacted."""

    masked_count: int
    chars_before: int
    chars_after: int
    chars_saved: int
    estimated_tokens_saved: int


def compact_messages(
    messages: list[dict[str, Any]],
    *,
    keep_recent_tool_outputs: int = 3,
    min_chars_to_mask: int = 350,
    in_place: bool = True,
) -> tuple[list[dict[str, Any]], CompactionStats]:
    """Deterministically mask older verbose tool outputs while preserving reasoning & recent state.

    Never modifies system messages, user messages, assistant messages, or the most recent
    `keep_recent_tool_outputs` tool responses.
    """
    target = messages if in_place else [dict(m) for m in messages]
    chars_before = sum(len(str(m.get("content") or "")) for m in target)

    tool_indices = [idx for idx, m in enumerate(target) if m.get("role") == "tool"]
    protected = (
        set(tool_indices[-keep_recent_tool_outputs:]) if keep_recent_tool_outputs > 0 else set()
    )

    masked_count = 0
    for idx in tool_indices:
        if idx in protected:
            continue
        content = str(target[idx].get("content") or "")
        if len(content) < min_chars_to_mask or content.startswith("[masked "):
            continue
        tool_name = str(target[idx].get("name") or "tool")
        first_line = content.splitlines()[0][:100] if content.splitlines() else ""
        placeholder = (
            f"[masked previous {tool_name} output: {len(content)} chars omitted "
            f"(preview: {first_line!r}) — call {tool_name} again if needed]"
        )
        target[idx]["content"] = placeholder
        masked_count += 1

    chars_after = sum(len(str(m.get("content") or "")) for m in target)
    chars_saved = max(0, chars_before - chars_after)
    return target, CompactionStats(
        masked_count=masked_count,
        chars_before=chars_before,
        chars_after=chars_after,
        chars_saved=chars_saved,
        estimated_tokens_saved=chars_saved // 4,
    )
