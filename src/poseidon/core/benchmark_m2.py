"""Milestone 2 benchmark: Whole-file edit vs. Unified-diff edit from identical copies."""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable
from pathlib import Path

from pydantic import BaseModel

from poseidon.core.diff_edit import run_diff_edit_request
from poseidon.core.single_file import CompletionClient, FileExtractionError, run_single_file_request
from poseidon.llm.openrouter import OpenRouterError
from poseidon.workspace import WorkspaceGuard


class BenchmarkRow(BaseModel):
    """Single row in the Milestone 2 whole-file vs. unified-diff comparison."""

    strategy: str
    model: str
    edit_applied: bool
    test_passed: bool
    test_detail: str
    attempts: int
    total_output_tokens: int | None = None
    elapsed_seconds: float
    cost_usd: float | None = None
    diff_text: str | None = None


DEFAULT_BENCHMARK_PROMPT = (
    "Add an optional --currency string flag (default '$') to ledger.py and prefix every printed "
    "monetary amount (category totals, grand total, budget, remaining, and over-budget amount) "
    "with the currency symbol. For example, with --currency 'EUR ', print 'food: EUR 20.00' and "
    "'Grand total: EUR 820.00'."
)


def default_currency_verifier(workspace_dir: Path, target_file: str) -> tuple[bool, str]:
    """Run the same verification test against the modified ledger.py."""
    script_path = workspace_dir / target_file
    cmd = [
        sys.executable,
        str(script_path),
        "--expenses",
        "food:12.50,rent:800,food:7.50",
        "--currency",
        "EUR ",
    ]
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return False, "Test timed out after 10s"

    output = proc.stdout.strip()
    if proc.returncode != 0:
        return False, f"Exit {proc.returncode}: {proc.stderr.strip()[:160]}"

    required_tokens = ["EUR 20.00", "EUR 800.00", "EUR 820.00"]
    missing = [tok for tok in required_tokens if tok not in output]
    if missing:
        return False, f"Missing expected output {missing}; got: {output!r}"
    return True, "PASS (verified EUR 20.00, EUR 800.00, EUR 820.00)"


def run_whole_vs_diff_benchmark(
    *,
    client: CompletionClient,
    source_workspace: Path,
    target_file: str,
    model: str,
    prompt: str = DEFAULT_BENCHMARK_PROMPT,
    max_diff_retries: int = 1,
    verifier: Callable[[Path, str], tuple[bool, str]] = default_currency_verifier,
) -> list[BenchmarkRow]:
    """Compare whole-file vs. unified-diff edits from identical clean copies of target_file."""
    source_path = (source_workspace / target_file).resolve()
    if not source_path.is_file():
        raise FileNotFoundError(f"Source demo file not found: {source_path}")

    rows: list[BenchmarkRow] = []

    with tempfile.TemporaryDirectory(prefix="poseidon_m2_") as tmp_dir:
        base_tmp = Path(tmp_dir)
        whole_dir = base_tmp / "whole"
        diff_dir = base_tmp / "diff"
        whole_dir.mkdir()
        diff_dir.mkdir()
        shutil.copy2(source_path, whole_dir / target_file)
        shutil.copy2(source_path, diff_dir / target_file)

        # 1. Whole-file strategy
        whole_started = time.perf_counter()
        try:
            whole_res = run_single_file_request(
                client=client,
                workspace=WorkspaceGuard(whole_dir),
                model=model,
                prompt=prompt,
                target_file=target_file,
            )
            whole_elapsed = time.perf_counter() - whole_started
            passed, detail = verifier(whole_dir, target_file)
            rows.append(
                BenchmarkRow(
                    strategy="whole-file",
                    model=whole_res.model,
                    edit_applied=True,
                    test_passed=passed,
                    test_detail=detail,
                    attempts=1,
                    total_output_tokens=whole_res.usage.completion_tokens,
                    elapsed_seconds=whole_elapsed,
                    cost_usd=whole_res.usage.cost_usd,
                )
            )
        except (OpenRouterError, FileExtractionError, ValueError) as exc:
            rows.append(
                BenchmarkRow(
                    strategy="whole-file",
                    model=model,
                    edit_applied=False,
                    test_passed=False,
                    test_detail=f"FAILED (edit not applied: {exc})",
                    attempts=1,
                    total_output_tokens=None,
                    elapsed_seconds=time.perf_counter() - whole_started,
                    cost_usd=None,
                )
            )

        # 2. Unified-diff strategy
        diff_res = run_diff_edit_request(
            client=client,
            workspace=WorkspaceGuard(diff_dir),
            model=model,
            prompt=prompt,
            target_file=target_file,
            max_retries=max_diff_retries,
            preview_and_confirm=None,
        )
        if not diff_res.applied:
            # Per assignment rule: if patch never applies, record failed edit even if old tests pass
            rows.append(
                BenchmarkRow(
                    strategy="unified-diff",
                    model=diff_res.model,
                    edit_applied=False,
                    test_passed=False,
                    test_detail=f"FAILED (edit not applied: {diff_res.reason})",
                    attempts=diff_res.attempts,
                    total_output_tokens=diff_res.total_output_tokens,
                    elapsed_seconds=diff_res.elapsed_seconds,
                    cost_usd=diff_res.total_cost_usd,
                    diff_text=diff_res.diff_text,
                )
            )
        else:
            passed, detail = verifier(diff_dir, target_file)
            rows.append(
                BenchmarkRow(
                    strategy="unified-diff",
                    model=diff_res.model,
                    edit_applied=True,
                    test_passed=passed,
                    test_detail=detail,
                    attempts=diff_res.attempts,
                    total_output_tokens=diff_res.total_output_tokens,
                    elapsed_seconds=diff_res.elapsed_seconds,
                    cost_usd=diff_res.total_cost_usd,
                    diff_text=diff_res.diff_text,
                )
            )

    return rows
