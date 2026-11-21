"""Milestone 1 single-file generation and full-file follow-up editing workflow."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Protocol

from pydantic import BaseModel

from poseidon.llm.openrouter import ChatResponse, TokenUsage
from poseidon.workspace import WorkspaceGuard

SYSTEM_PROMPT = """You are Poseidon-CLI, a precise Python software engineering assistant.
Your job is to create or update a SINGLE runnable Python source file inside the user's workspace.

Rules:
1. Return the COMPLETE, runnable Python source file (never omit code or use placeholders).
2. Wrap the output in this exact format:
<file path="filename.py">
# full python source code here
</file>
3. Use a simple relative filename (no leading slash, no '..').
4. Include clean English identifiers, docstrings, and comments.
5. Keep any explanation brief and outside the <file> tags.
"""

_FILE_TAG_RE = re.compile(
    r'<file\s+path=["\']([^"\']+)["\']\s*>\s*\n?(.*?)\n?</file>',
    re.DOTALL | re.IGNORECASE,
)
_CODE_FENCE_RE = re.compile(r"```(?:python|py)?\s*\n(.*?)\n```", re.DOTALL)


class CompletionClient(Protocol):
    """Protocol for any LLM client capable of chat completions."""

    def chat_completion(
        self,
        *,
        model: str,
        messages: list[dict[str, object]],
        temperature: float = 0.2,
    ) -> ChatResponse: ...


class FileExtractionError(ValueError):
    """Raised when a valid source file cannot be extracted from the model output."""


class SingleFileResult(BaseModel):
    """Outcome of a single-file generation or follow-up edit."""

    relative_path: str
    written_path: Path
    content: str
    model: str
    usage: TokenUsage
    elapsed_seconds: float
    is_update: bool


def extract_file_payload(
    raw_response: str,
    fallback_filename: str | None = None,
) -> tuple[str, str]:
    """Parse a (relative_path, source_code) pair from model output."""
    match = _FILE_TAG_RE.search(raw_response)
    if match:
        rel_path = match.group(1).strip()
        code = match.group(2)
        if not code.strip():
            raise FileExtractionError("Model returned an empty <file> block.")
        return rel_path, _ensure_trailing_newline(code)

    fence_match = _CODE_FENCE_RE.search(raw_response)
    if fence_match and fallback_filename:
        code = fence_match.group(1)
        if not code.strip():
            raise FileExtractionError("Model returned an empty fenced code block.")
        return fallback_filename, _ensure_trailing_newline(code)

    raise FileExtractionError(
        "Could not extract a <file path='...'>...</file> block from the model response."
    )


def _ensure_trailing_newline(text: str) -> str:
    return text if text.endswith("\n") else f"{text}\n"


def run_single_file_request(
    *,
    client: CompletionClient,
    workspace: WorkspaceGuard,
    model: str,
    prompt: str,
    target_file: str | None = None,
) -> SingleFileResult:
    """Create or update one runnable source file inside the sandboxed workspace."""
    existing_files = workspace.list_files()
    resolved_target = target_file
    if resolved_target is None and len(existing_files) == 1:
        resolved_target = existing_files[0]

    existing_content: str | None = None
    if resolved_target is not None:
        try:
            existing_content = workspace.read_text(resolved_target)
        except FileNotFoundError:
            existing_content = None

    user_message_parts: list[str] = []
    if existing_content is not None and resolved_target is not None:
        user_message_parts.append(
            f"Current workspace file `{resolved_target}`:\n```python\n{existing_content}```"
        )
        user_message_parts.append(
            f"Follow-up request (update `{resolved_target}` and return the full file):\n{prompt}"
        )
    else:
        if resolved_target:
            user_message_parts.append(f"Target filename: `{resolved_target}`")
        user_message_parts.append(f"Request:\n{prompt}")

    messages: list[dict[str, object]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "\n\n".join(user_message_parts)},
    ]

    response = client.chat_completion(model=model, messages=messages)
    rel_path, code = extract_file_payload(
        response.content,
        fallback_filename=resolved_target or "main.py",
    )

    written_path = workspace.write_text(rel_path, code)
    return SingleFileResult(
        relative_path=rel_path,
        written_path=written_path,
        content=code,
        model=response.model,
        usage=response.usage,
        elapsed_seconds=response.elapsed_seconds,
        is_update=existing_content is not None,
    )
