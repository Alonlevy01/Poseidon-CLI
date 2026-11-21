"""Unit tests for Milestone 1 single-file creation and follow-up edits."""

from pathlib import Path

import pytest

from poseidon.core.single_file import (
    FileExtractionError,
    extract_file_payload,
    run_single_file_request,
)
from poseidon.llm.openrouter import ChatResponse, TokenUsage
from poseidon.workspace import WorkspaceGuard, WorkspaceSecurityError


class FakeClient:
    """Deterministic fake OpenRouter client for zero-cost unit testing."""

    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self.calls: list[list[dict[str, object]]] = []

    def chat_completion(
        self,
        *,
        model: str,
        messages: list[dict[str, object]],
        temperature: float = 0.2,
    ) -> ChatResponse:
        self.calls.append(messages)
        content = self._responses.pop(0)
        return ChatResponse(
            model=model,
            content=content,
            usage=TokenUsage(
                prompt_tokens=100,
                completion_tokens=50,
                total_tokens=150,
                cost_usd=0.0001,
            ),
            elapsed_seconds=0.12,
            finish_reason="stop",
        )


def test_extract_file_payload_from_file_tag() -> None:
    raw = 'Here is the program:\n<file path="ledger.py">\nprint(42)\n</file>\nDone.'
    rel_path, code = extract_file_payload(raw)
    assert rel_path == "ledger.py"
    assert code == "print(42)\n"


def test_extract_file_payload_from_fence_fallback() -> None:
    raw = "```python\nx = 1 + 2\n```"
    rel_path, code = extract_file_payload(raw, fallback_filename="calc.py")
    assert rel_path == "calc.py"
    assert code == "x = 1 + 2\n"


def test_extract_file_payload_raises_on_unparseable() -> None:
    with pytest.raises(FileExtractionError):
        extract_file_payload("No code here at all.")


def test_create_and_followup_whole_file_edit(tmp_path: Path) -> None:
    guard = WorkspaceGuard(tmp_path)
    client = FakeClient(
        [
            '<file path="ledger.py">\ndef total(items):\n    return sum(items)\n</file>',
            '<file path="ledger.py">\ndef total(items):\n    return round(sum(items), 2)\n</file>',
        ]
    )

    first = run_single_file_request(
        client=client,
        workspace=guard,
        model="openai/gpt-4o-mini",
        prompt="Create a ledger file with a total function.",
    )
    assert not first.is_update
    assert first.relative_path == "ledger.py"
    assert "return sum(items)" in guard.read_text("ledger.py")

    second = run_single_file_request(
        client=client,
        workspace=guard,
        model="openai/gpt-4o-mini",
        prompt="Round the total to 2 decimal places.",
    )
    assert second.is_update
    assert "round(sum(items), 2)" in guard.read_text("ledger.py")
    # Verify the existing file content was included in the follow-up prompt
    followup_user_msg = str(client.calls[1][1]["content"])
    assert "return sum(items)" in followup_user_msg


def test_model_cannot_escape_workspace_via_file_tag(tmp_path: Path) -> None:
    guard = WorkspaceGuard(tmp_path)
    client = FakeClient(['<file path="../evil.py">\nprint("escaped")\n</file>'])

    with pytest.raises(WorkspaceSecurityError):
        run_single_file_request(
            client=client,
            workspace=guard,
            model="openai/gpt-4o-mini",
            prompt="Write a script.",
        )
