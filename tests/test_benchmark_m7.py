"""Unit tests for Milestone 7 model validation and 2-model comparison with acceptance tests."""

from pathlib import Path
from typing import Any

import httpx
import pytest

from poseidon.config import Settings
from poseidon.core.benchmark_m7 import run_model_comparison
from poseidon.llm.openrouter import (
    ChatResponse,
    ModelToolSupportError,
    OpenRouterClient,
    OpenRouterError,
    TokenUsage,
)
from poseidon.sandbox.docker_runner import TestFailure, TestRunResult


def test_validate_tool_model_checks_availability_and_tool_support() -> None:
    def _mock_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "data": [
                    {
                        "id": "nvidia/nemotron-3.5-super-vl:free",
                        "supported_parameters": ["temperature", "tools"],
                    },
                    {
                        "id": "demo/no-tools-model:free",
                        "supported_parameters": ["temperature"],
                    },
                ]
            },
        )

    transport = httpx.MockTransport(_mock_handler)
    http = httpx.Client(base_url="https://x", transport=transport)
    with OpenRouterClient(Settings(), http_client=http) as client:
        info = client.validate_tool_model("nvidia/nemotron-3.5-super-vl:free")
        assert info["id"] == "nvidia/nemotron-3.5-super-vl:free"

        with pytest.raises(ModelToolSupportError, match="does not support native tool calling"):
            client.validate_tool_model("demo/no-tools-model:free")

        with pytest.raises(ModelToolSupportError, match="unavailable or not found"):
            client.validate_tool_model("nonexistent/model:free")


def test_model_comparison_runs_both_models_and_records_failures(tmp_path: Path) -> None:
    base = tmp_path / "base"
    base.mkdir()
    (base / "ledger.py").write_text("print('base')\n", encoding="utf-8")
    (base / "test_ledger.py").write_text("def test_ok(): assert True\n", encoding="utf-8")

    acceptance = tmp_path / "test_acc.py"
    acceptance.write_text("def test_acc(): assert True\n", encoding="utf-8")

    class MultiModelFakeClient:
        def chat_completion(
            self,
            *,
            model: str,
            messages: list[dict[str, Any]],
            temperature: float = 0.2,
            tools: list[dict[str, Any]] | None = None,
            tool_choice: str | None = None,
        ) -> ChatResponse:
            if model == "family-b/broken:free":
                raise OpenRouterError("Provider rate limit")
            return ChatResponse(
                model=model,
                content="Completed edit.",
                tool_calls=[],
                usage=TokenUsage(
                    prompt_tokens=100,
                    completion_tokens=40,
                    total_tokens=140,
                    cost_usd=0.0,
                ),
                elapsed_seconds=0.1,
                finish_reason="stop",
            )

    class FakeSandboxRunner:
        @property
        def timeout_seconds(self) -> int:
            return 15

        def run(
            self,
            workspace_root: Path,
            *,
            test_args: list[str] | None = None,
            extra_files: dict[str, str] | None = None,
        ) -> TestRunResult:
            if extra_files:
                # Acceptance test fails on unedited base code
                return TestRunResult(
                    command="pytest test_acc.py",
                    timeout_seconds=15,
                    exit_code=1,
                    passed=0,
                    failed=1,
                    total=1,
                    failures=[TestFailure(name="test_acc", message="missing --exclude")],
                    verified=False,
                    unverified_reason="1 test(s) failed",
                )
            return TestRunResult(
                command="pytest -v --tb=short",
                timeout_seconds=15,
                exit_code=0,
                passed=1,
                failed=0,
                total=1,
                verified=True,
            )

    rows = run_model_comparison(
        client=MultiModelFakeClient(),
        models=["family-a/good:free", "family-b/broken:free"],
        base_workspace=base,
        acceptance_test_path=acceptance,
        test_runner=FakeSandboxRunner(),
        max_calls=4,
    )

    assert len(rows) == 2
    assert rows[0].model_id == "family-a/good:free"
    assert rows[0].visible_tests_passed is True
    assert rows[0].acceptance_passed is False
    assert "FAIL" in rows[0].acceptance_summary

    assert rows[1].model_id == "family-b/broken:free"
    assert rows[1].visible_tests_passed is False
    assert rows[1].acceptance_passed is False
    assert rows[1].error is not None
