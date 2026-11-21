"""Exercise 1 Optional Challenge: Before/After benchmark of Python AST preflight guard
on both the original multi-file task (--exclude) and a held-out second task (--min-amount).
"""

from __future__ import annotations

import shutil
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel

from poseidon.core.agent import ToolCapableClient, run_agent_loop
from poseidon.core.benchmark_m7 import DEFAULT_M5_REQUEST, _summarize_test
from poseidon.llm.openrouter import OpenRouterError
from poseidon.observability.respan import PoseidonTracer
from poseidon.sandbox.docker_runner import TestRunner
from poseidon.tools.builtin import register_core_tools
from poseidon.tools.registry import ToolRegistry, ToolSessionState
from poseidon.workspace import WorkspaceGuard

HELDOUT_MIN_AMOUNT_REQUEST = (
    "Add an optional --min-amount float flag that ignores categories whose total is strictly "
    "below the threshold before computing the grand total and any budget comparison. Keep "
    "threshold filtering in the analytics module, summary-line formatting in the formatting "
    "module, and wire the CLI flag in the entrypoint so that when categories are filtered out, "
    "an extra line prints listing those filtered category names sorted alphabetically, "
    "for example 'Filtered out: coffee, snacks'."
)


@dataclass(frozen=True)
class ChallengeTask:
    """One coding task + its held-out acceptance test for before/after harness comparison."""

    name: str
    request: str
    acceptance_test_path: Path


class ChallengeRunRow(BaseModel):
    """One row of the v1 (no preflight) vs. v2 (+AST preflight) challenge comparison."""

    harness_version: str
    task_name: str
    visible_passed: bool
    visible_summary: str
    acceptance_passed: bool
    acceptance_summary: str
    calls_made: int
    max_calls: int
    elapsed_seconds: float
    total_tokens: int | None = None


def _copy_clean(source_dir: Path, dest_dir: Path) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    guard = WorkspaceGuard(source_dir)
    for rel in guard.list_files():
        src = source_dir / rel
        dst = dest_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def run_optional_challenge_benchmark(
    *,
    client: ToolCapableClient,
    model: str,
    base_workspace: Path,
    exclude_acceptance_path: Path,
    min_amount_acceptance_path: Path,
    test_runner: TestRunner,
    max_calls: int = 10,
    tracer: PoseidonTracer | None = None,
    on_step: Callable[[str, str], None] | None = None,
) -> list[ChallengeRunRow]:
    """Run v1 (preflight=False) and v2 (preflight=True) on original + held-out tasks."""
    tasks = [
        ChallengeTask(
            name="Task 1 (original --exclude)",
            request=DEFAULT_M5_REQUEST,
            acceptance_test_path=exclude_acceptance_path,
        ),
        ChallengeTask(
            name="Task 2 (held-out --min-amount)",
            request=HELDOUT_MIN_AMOUNT_REQUEST,
            acceptance_test_path=min_amount_acceptance_path,
        ),
    ]
    versions = [
        ("v1 (baseline)", False),
        ("v2 (+AST preflight)", True),
    ]

    rows: list[ChallengeRunRow] = []
    with tempfile.TemporaryDirectory(prefix="poseidon_challenge_") as tmp_str:
        tmp_root = Path(tmp_str)
        run_num = 0
        for task in tasks:
            acc_code = task.acceptance_test_path.read_text(encoding="utf-8")
            acc_name = task.acceptance_test_path.name

            for version_label, preflight_enabled in versions:
                run_num += 1
                if on_step is not None:
                    on_step(version_label, task.name)

                run_dir = tmp_root / f"run_{run_num}"
                _copy_clean(base_workspace, run_dir)
                guard = WorkspaceGuard(run_dir)
                state = ToolSessionState()
                registry = ToolRegistry()
                register_core_tools(
                    registry=registry,
                    workspace=guard,
                    state=state,
                    confirm_patch=None,
                    test_runner=test_runner,
                    protect_tests=True,
                    enable_preflight=preflight_enabled,
                )

                started = time.perf_counter()
                try:
                    agent_res = run_agent_loop(
                        client=client,
                        registry=registry,
                        model=model,
                        user_prompt=task.request,
                        max_calls=max_calls,
                        tracer=tracer,
                        workspace_root=guard.root,
                        tool_state=state,
                        test_runner=test_runner,
                    )
                    vis_ok, vis_sum = _summarize_test(test_runner.run(run_dir), label="visible")
                    acc_res = test_runner.run(
                        run_dir,
                        test_args=[acc_name],
                        extra_files={acc_name: acc_code},
                    )
                    acc_ok, acc_sum = _summarize_test(acc_res, label="acceptance")
                    rows.append(
                        ChallengeRunRow(
                            harness_version=version_label,
                            task_name=task.name,
                            visible_passed=vis_ok,
                            visible_summary=vis_sum,
                            acceptance_passed=acc_ok,
                            acceptance_summary=acc_sum,
                            calls_made=agent_res.calls_made,
                            max_calls=max_calls,
                            elapsed_seconds=agent_res.elapsed_seconds,
                            total_tokens=agent_res.usage.total_tokens,
                        )
                    )
                except (OpenRouterError, ValueError, RuntimeError) as exc:
                    rows.append(
                        ChallengeRunRow(
                            harness_version=version_label,
                            task_name=task.name,
                            visible_passed=False,
                            visible_summary=f"ERROR ({str(exc)[:40]})",
                            acceptance_passed=False,
                            acceptance_summary="FAIL (aborted)",
                            calls_made=0,
                            max_calls=max_calls,
                            elapsed_seconds=time.perf_counter() - started,
                            total_tokens=None,
                        )
                    )

    return rows
