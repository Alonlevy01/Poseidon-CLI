"""Milestone 7: Compare two tool-capable OpenRouter models from different families
on the Milestone 5 multi-file request, evaluated with a hidden acceptance test.
"""

from __future__ import annotations

import shutil
import tempfile
import time
from collections.abc import Callable
from pathlib import Path

from pydantic import BaseModel

from poseidon.core.agent import ToolCapableClient, run_agent_loop
from poseidon.indexing.semantic_index import EmbeddingProvider, SemanticCodeIndex
from poseidon.llm.openrouter import OpenRouterError
from poseidon.observability.respan import PoseidonTracer
from poseidon.sandbox.docker_runner import TestRunner, TestRunResult
from poseidon.tools.builtin import register_core_tools
from poseidon.tools.registry import ToolRegistry, ToolSessionState
from poseidon.workspace import WorkspaceGuard

DEFAULT_M5_REQUEST = (
    "Add an optional --exclude flag that accepts a comma-separated list of categories to ignore "
    "before computing category totals, the grand total, and any budget comparison. Keep category "
    "filtering in the analytics module, excluded-line formatting in the formatting module, and "
    "wire the CLI option in the entrypoint so that when one or more categories are excluded, "
    "an extra line like 'Excluded: entertainment, snacks' (sorted alphabetically) prints in "
    "the report."
)


class ModelComparisonRow(BaseModel):
    """Summary row for one model evaluated in the Milestone 7 comparison."""

    model_id: str
    visible_tests_passed: bool
    visible_tests_summary: str
    acceptance_passed: bool
    acceptance_summary: str
    calls_made: int
    max_calls: int
    stop_reason: str
    modified_files: list[str]
    elapsed_seconds: float
    output_tokens: int | None = None
    total_tokens: int | None = None
    cost_usd: float | None = None
    error: str | None = None


def _summarize_test(res: TestRunResult | None, *, label: str) -> tuple[bool, str]:
    if res is None:
        return False, f"FAIL ({label} did not run)"
    if res.verified:
        return True, f"PASS ({res.passed}/{res.total})"
    if res.timed_out:
        return False, f"TIMEOUT ({res.timeout_seconds}s)"
    if res.runner_error:
        return False, f"ERROR ({res.runner_error[:60]})"
    if res.total == 0:
        return False, "FAIL (0 tests ran)"
    first_err = res.failures[0].message.splitlines()[0][:60] if res.failures else ""
    detail = f" — {first_err}" if first_err else ""
    return False, f"FAIL ({res.passed}/{res.total} passed{detail})"


def _copy_clean_workspace(source_dir: Path, dest_dir: Path) -> None:
    """Copy only clean non-excluded source/test files into an isolated workspace."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    src_guard = WorkspaceGuard(source_dir)
    for rel_str in src_guard.list_files():
        src_file = source_dir / rel_str
        dst_file = dest_dir / rel_str
        dst_file.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_file, dst_file)


def run_model_comparison(
    *,
    client: ToolCapableClient,
    models: list[str],
    base_workspace: Path,
    acceptance_test_path: Path,
    test_runner: TestRunner,
    embedding_provider: EmbeddingProvider | None = None,
    request: str = DEFAULT_M5_REQUEST,
    max_calls: int = 12,
    tracer: PoseidonTracer | None = None,
    on_model_start: Callable[[str], None] | None = None,
) -> list[ModelComparisonRow]:
    """Run each model on an identical clean copy, then run the hidden acceptance test on both."""
    if not base_workspace.is_dir():
        raise FileNotFoundError(f"Base workspace revision not found: {base_workspace}")
    if not acceptance_test_path.is_file():
        raise FileNotFoundError(f"Acceptance test file not found: {acceptance_test_path}")

    acceptance_code = acceptance_test_path.read_text(encoding="utf-8")
    acceptance_name = acceptance_test_path.name

    with tempfile.TemporaryDirectory(prefix="poseidon_m7_") as tmp_root_str:
        tmp_root = Path(tmp_root_str)
        saved_runs: list[tuple[str, Path, ModelComparisonRow | None, object | None]] = []

        # Phase 1: Run EVERY model from a clean copy BEFORE touching the hidden acceptance test
        for idx, model_id in enumerate(models, start=1):
            if on_model_start is not None:
                on_model_start(model_id)

            run_dir = tmp_root / f"model_{idx}"
            _copy_clean_workspace(base_workspace, run_dir)
            guard = WorkspaceGuard(run_dir)
            state = ToolSessionState()
            registry = ToolRegistry()

            sem_index = (
                SemanticCodeIndex(guard, embedding_provider)
                if embedding_provider is not None
                else None
            )
            register_core_tools(
                registry=registry,
                workspace=guard,
                state=state,
                confirm_patch=None,  # Auto-approve in isolated temp comparison copy
                test_runner=test_runner,
                semantic_index=sem_index,
                protect_tests=True,
            )

            started = time.perf_counter()
            try:
                agent_res = run_agent_loop(
                    client=client,
                    registry=registry,
                    model=model_id,
                    user_prompt=request,
                    max_calls=max_calls,
                    tracer=tracer,
                    workspace_root=guard.root,
                    tool_state=state,
                    test_runner=test_runner,
                )
                saved_runs.append((model_id, run_dir, None, agent_res))
            except (OpenRouterError, ValueError, RuntimeError) as exc:
                failed_row = ModelComparisonRow(
                    model_id=model_id,
                    visible_tests_passed=False,
                    visible_tests_summary="ERROR (run aborted)",
                    acceptance_passed=False,
                    acceptance_summary="FAIL (run aborted)",
                    calls_made=0,
                    max_calls=max_calls,
                    stop_reason=f"error: {exc}",
                    modified_files=[],
                    elapsed_seconds=time.perf_counter() - started,
                    error=str(exc),
                )
                saved_runs.append((model_id, run_dir, failed_row, None))

        # Phase 2: After BOTH runs stop, evaluate visible + hidden acceptance tests in sandbox
        rows: list[ModelComparisonRow] = []
        for _model_id, run_dir, failed_row, agent_res in saved_runs:
            if failed_row is not None or agent_res is None:
                assert failed_row is not None
                rows.append(failed_row)
                continue

            visible_res = test_runner.run(run_dir)
            vis_ok, vis_summary = _summarize_test(visible_res, label="visible tests")

            accept_res = test_runner.run(
                run_dir,
                test_args=[acceptance_name],
                extra_files={acceptance_name: acceptance_code},
            )
            acc_ok, acc_summary = _summarize_test(accept_res, label="acceptance test")

            rows.append(
                ModelComparisonRow(
                    model_id=agent_res.model,
                    visible_tests_passed=vis_ok,
                    visible_tests_summary=vis_summary,
                    acceptance_passed=acc_ok,
                    acceptance_summary=acc_summary,
                    calls_made=agent_res.calls_made,
                    max_calls=agent_res.max_calls,
                    stop_reason=agent_res.stop_reason,
                    modified_files=agent_res.modified_files,
                    elapsed_seconds=agent_res.elapsed_seconds,
                    output_tokens=agent_res.usage.completion_tokens,
                    total_tokens=agent_res.usage.total_tokens,
                    cost_usd=agent_res.usage.cost_usd,
                )
            )

        return rows
