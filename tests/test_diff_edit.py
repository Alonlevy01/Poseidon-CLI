"""Unit tests for Milestone 2 diff-edit workflow and whole-file vs. diff benchmark."""

from pathlib import Path

from poseidon.core.benchmark_m2 import run_whole_vs_diff_benchmark
from poseidon.core.diff_edit import run_diff_edit_request
from poseidon.llm.openrouter import ChatResponse, TokenUsage
from poseidon.workspace import WorkspaceGuard


class SequenceClient:
    """Mock client returning scripted responses and token counts."""

    def __init__(self, responses: list[tuple[str, int]]) -> None:
        self._responses = list(responses)

    def chat_completion(
        self,
        *,
        model: str,
        messages: list[dict[str, object]],
        temperature: float = 0.2,
    ) -> ChatResponse:
        content, out_tokens = self._responses.pop(0)
        return ChatResponse(
            model=model,
            content=content,
            usage=TokenUsage(
                prompt_tokens=120,
                completion_tokens=out_tokens,
                total_tokens=120 + out_tokens,
                cost_usd=0.0,
            ),
            elapsed_seconds=0.05,
            finish_reason="stop",
        )


BASE_SCRIPT = """def price(x: float) -> str:
    return f"{x:.2f}"


if __name__ == "__main__":
    print(price(20))
"""


def test_diff_edit_retries_after_bad_patch_and_applies(tmp_path: Path) -> None:
    guard = WorkspaceGuard(tmp_path)
    guard.write_text("demo.py", BASE_SCRIPT)

    bad_resp = (
        "<diff>\n--- a/demo.py\n+++ b/demo.py\n@@ -1,2 +1,2 @@\n"
        "-    return 'wrong'\n+    return f'EUR {x:.2f}'\n</diff>"
    )
    good_resp = (
        "<diff>\n--- a/demo.py\n+++ b/demo.py\n@@ -1,2 +1,2 @@\n"
        ' def price(x: float) -> str:\n-    return f"{x:.2f}"\n+    return f"EUR {x:.2f}"\n</diff>'
    )
    client = SequenceClient([(bad_resp, 40), (good_resp, 45)])

    result = run_diff_edit_request(
        client=client,
        workspace=guard,
        model="nvidia/nemotron-3.5-super-vl:free",
        prompt="Prefix price with EUR ",
        target_file="demo.py",
        max_retries=1,
    )
    assert result.applied
    assert result.attempts == 2
    assert result.total_output_tokens == 85
    assert 'EUR {x:.2f}' in guard.read_text("demo.py")


def test_diff_edit_user_rejection_leaves_file_untouched(tmp_path: Path) -> None:
    guard = WorkspaceGuard(tmp_path)
    guard.write_text("demo.py", BASE_SCRIPT)

    good_resp = (
        "<diff>\n--- a/demo.py\n+++ b/demo.py\n@@ -1,2 +1,2 @@\n"
        ' def price(x: float) -> str:\n-    return f"{x:.2f}"\n+    return f"EUR {x:.2f}"\n</diff>'
    )
    client = SequenceClient([(good_resp, 45)])

    result = run_diff_edit_request(
        client=client,
        workspace=guard,
        model="nvidia/nemotron-3.5-super-vl:free",
        prompt="Prefix price with EUR ",
        target_file="demo.py",
        preview_and_confirm=lambda _diff: False,
    )
    assert not result.applied
    assert result.rejected_by_user
    assert guard.read_text("demo.py") == BASE_SCRIPT


def test_benchmark_records_failed_edit_when_patch_never_applies(tmp_path: Path) -> None:
    guard = WorkspaceGuard(tmp_path)
    guard.write_text("demo.py", BASE_SCRIPT)

    whole_resp = (
        '<file path="demo.py">\ndef price(x: float) -> str:\n    return f"EUR {x:.2f}"\n</file>'
    )
    bad_diff = (
        "<diff>\n--- a/demo.py\n+++ b/demo.py\n@@ -1,2 +1,2 @@\n"
        "-    nonexistent()\n+    return 'x'\n</diff>"
    )
    client = SequenceClient([(whole_resp, 200), (bad_diff, 35), (bad_diff, 35)])

    rows = run_whole_vs_diff_benchmark(
        client=client,
        source_workspace=tmp_path,
        target_file="demo.py",
        model="nvidia/nemotron-3.5-super-vl:free",
        prompt="Prefix with EUR",
        max_diff_retries=1,
        verifier=lambda _ws, _f: (True, "PASS"),
    )

    assert len(rows) == 2
    assert rows[0].strategy == "whole-file"
    assert rows[0].edit_applied and rows[0].test_passed
    assert rows[1].strategy == "unified-diff"
    assert not rows[1].edit_applied
    # Even though verifier returns True, failed patch MUST record test_passed=False
    assert not rows[1].test_passed
    assert rows[1].total_output_tokens == 70
