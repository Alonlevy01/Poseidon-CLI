"""Milestone 2 unified-diff editing workflow with preview, validation, and retry metrics."""

from __future__ import annotations

import time
from collections.abc import Callable

from pydantic import BaseModel

from poseidon.core.single_file import CompletionClient
from poseidon.llm.openrouter import TokenUsage
from poseidon.patching.unified_diff import (
    PatchError,
    apply_unified_diff,
    extract_diff_payload,
    sha256_text,
)
from poseidon.workspace import WorkspaceGuard

DIFF_SYSTEM_PROMPT = """You are Poseidon-CLI, a precise Python software engineering assistant.
Your job is to edit an existing Python file by returning ONLY a valid Git unified diff.

Rules:
1. Return ONLY the minimal unified diff (removed lines with '-', added lines with '+',
   and 3 lines of unchanged context with a leading space).
2. Do NOT return the whole file.
3. Wrap the unified diff inside <diff>...</diff> tags with standard headers:
<diff>
--- a/filename.py
+++ b/filename.py
@@ -10,6 +10,7 @@
 unchanged line
-old line
+new line
 unchanged line
</diff>
4. Context lines and removed lines MUST match the current file character-for-character.
"""


class DiffEditResult(BaseModel):
    """Outcome and metrics for a unified-diff edit request."""

    applied: bool
    rejected_by_user: bool = False
    reason: str | None = None
    diff_text: str | None = None
    modified_files: list[str] = []
    added_lines: int = 0
    removed_lines: int = 0
    model: str
    attempts: int
    total_output_tokens: int | None = None
    total_tokens: int | None = None
    total_cost_usd: float | None = None
    elapsed_seconds: float


def _accumulate_usage(usages: list[TokenUsage]) -> tuple[int | None, int | None, float | None]:
    """Sum token counts and costs across attempts, returning None if unavailable."""
    out_vals = [u.completion_tokens for u in usages if u.completion_tokens is not None]
    tot_vals = [u.total_tokens for u in usages if u.total_tokens is not None]
    cost_vals = [u.cost_usd for u in usages if u.cost_usd is not None]
    return (
        sum(out_vals) if out_vals else None,
        sum(tot_vals) if tot_vals else None,
        sum(cost_vals) if cost_vals else None,
    )


def _numbered_source(content: str) -> str:
    """Format file contents with 1-based line numbers to improve model hunk accuracy."""
    return "\n".join(f"{idx:4d} | {line}" for idx, line in enumerate(content.splitlines(), start=1))


def run_diff_edit_request(
    *,
    client: CompletionClient,
    workspace: WorkspaceGuard,
    model: str,
    prompt: str,
    target_file: str,
    max_retries: int = 1,
    preview_and_confirm: Callable[[str], bool] | None = None,
) -> DiffEditResult:
    """Request a unified diff from the model, preview it, verify base hash, and apply atomically."""
    original_text = workspace.read_text(target_file)
    expected_hashes = {target_file: sha256_text(original_text)}

    messages: list[dict[str, object]] = [
        {"role": "system", "content": DIFF_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Current file `{target_file}` (with line numbers for reference only; "
                f"do NOT include line numbers in the diff):\n"
                f"```python\n{_numbered_source(original_text)}\n```\n\n"
                f"Requested change:\n{prompt}"
            ),
        },
    ]

    usages: list[TokenUsage] = []
    started = time.perf_counter()
    last_error: str | None = None
    last_diff: str | None = None
    resolved_model = model

    for attempt in range(1, max_retries + 2):
        response = client.chat_completion(model=model, messages=messages)
        resolved_model = response.model
        usages.append(response.usage)

        try:
            diff_text = extract_diff_payload(response.content)
            last_diff = diff_text
            # Dry-run validate base hash + full patch before preview/apply
            apply_unified_diff(
                workspace=workspace,
                diff_text=diff_text,
                expected_hashes=expected_hashes,
                dry_run=True,
            )
        except PatchError as exc:
            last_error = str(exc)
            if attempt <= max_retries:
                messages.append({"role": "assistant", "content": response.content})
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            f"Your unified diff failed validation: {last_error}\n"
                            "Please return a corrected unified diff inside <diff>...</diff> tags "
                            "matching the exact lines of the file."
                        ),
                    }
                )
                continue
            out_tok, tot_tok, cost = _accumulate_usage(usages)
            return DiffEditResult(
                applied=False,
                reason=last_error,
                diff_text=last_diff,
                model=resolved_model,
                attempts=attempt,
                total_output_tokens=out_tok,
                total_tokens=tot_tok,
                total_cost_usd=cost,
                elapsed_seconds=time.perf_counter() - started,
            )

        # Patch validated cleanly — preview & ask for confirmation if callback provided
        if preview_and_confirm is not None and not preview_and_confirm(diff_text):
            out_tok, tot_tok, cost = _accumulate_usage(usages)
            return DiffEditResult(
                applied=False,
                rejected_by_user=True,
                reason="User declined to apply the patch.",
                diff_text=diff_text,
                model=resolved_model,
                attempts=attempt,
                total_output_tokens=out_tok,
                total_tokens=tot_tok,
                total_cost_usd=cost,
                elapsed_seconds=time.perf_counter() - started,
            )

        # Apply for real (re-checking base hash in case file changed during preview!)
        try:
            patch_result = apply_unified_diff(
                workspace=workspace,
                diff_text=diff_text,
                expected_hashes=expected_hashes,
                dry_run=False,
            )
        except PatchError as exc:
            out_tok, tot_tok, cost = _accumulate_usage(usages)
            return DiffEditResult(
                applied=False,
                reason=str(exc),
                diff_text=diff_text,
                model=resolved_model,
                attempts=attempt,
                total_output_tokens=out_tok,
                total_tokens=tot_tok,
                total_cost_usd=cost,
                elapsed_seconds=time.perf_counter() - started,
            )

        out_tok, tot_tok, cost = _accumulate_usage(usages)
        return DiffEditResult(
            applied=True,
            diff_text=diff_text,
            modified_files=patch_result.modified_files,
            added_lines=patch_result.added_lines,
            removed_lines=patch_result.removed_lines,
            model=resolved_model,
            attempts=attempt,
            total_output_tokens=out_tok,
            total_tokens=tot_tok,
            total_cost_usd=cost,
            elapsed_seconds=time.perf_counter() - started,
        )

    out_tok, tot_tok, cost = _accumulate_usage(usages)
    return DiffEditResult(
        applied=False,
        reason=last_error or "Unknown diff error.",
        diff_text=last_diff,
        model=resolved_model,
        attempts=max_retries + 1,
        total_output_tokens=out_tok,
        total_tokens=tot_tok,
        total_cost_usd=cost,
        elapsed_seconds=time.perf_counter() - started,
    )
