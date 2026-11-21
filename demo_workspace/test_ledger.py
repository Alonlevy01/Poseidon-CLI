"""Unit tests for the demo Expense & Budget Ledger CLI (ledger.py)."""

import argparse
import subprocess
import sys
from pathlib import Path

import pytest

import ledger

LEDGER_SCRIPT = Path(__file__).parent / "ledger.py"


def _run_ledger(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(LEDGER_SCRIPT), *args],
        capture_output=True,
        text=True,
        check=False,
    )


def test_parse_expenses_groups_and_sums_categories() -> None:
    totals = ledger.parse_expenses("food:12.50,rent:800,food:7.50")
    assert totals == {"food": pytest.approx(20.0), "rent": pytest.approx(800.0)}


def test_parse_expenses_rejects_non_positive_amounts() -> None:
    with pytest.raises(argparse.ArgumentTypeError, match="positive"):
        ledger.parse_expenses("food:-5.00")
    with pytest.raises(argparse.ArgumentTypeError, match="positive"):
        ledger.parse_expenses("food:0")


def test_cli_prints_alphabetical_totals_and_grand_total() -> None:
    proc = _run_ledger("--expenses", "rent:800,food:12.50,food:7.50")
    assert proc.returncode == 0
    lines = proc.stdout.strip().splitlines()
    assert lines == [
        "food: $20.00",
        "rent: $800.00",
        "Grand total: $820.00",
    ]


def test_cli_budget_under_and_over_limit() -> None:
    under = _run_ledger("--expenses", "food:12.50,rent:800,food:7.50", "--budget", "900")
    assert under.returncode == 0
    assert "Budget: $900.00" in under.stdout
    assert "Remaining: $80.00" in under.stdout

    over = _run_ledger("--expenses", "food:12.50,rent:800,food:7.50", "--budget", "800")
    assert over.returncode == 0
    assert "Budget: $800.00" in over.stdout
    assert "OVER BUDGET by $20.00" in over.stdout


def test_cli_custom_currency_prefix() -> None:
    proc = _run_ledger("--expenses", "food:12.50,rent:800,food:7.50", "--currency", "EUR ")
    assert proc.returncode == 0
    assert "food: EUR 20.00" in proc.stdout
    assert "rent: EUR 800.00" in proc.stdout
    assert "Grand total: EUR 820.00" in proc.stdout


def test_cli_sort_by_amount_descending() -> None:
    proc = _run_ledger("--expenses", "food:12.50,rent:800,food:7.50", "--sort", "amount")
    assert proc.returncode == 0
    lines = proc.stdout.strip().splitlines()
    assert lines[0] == "rent: $800.00"
    assert lines[1] == "food: $20.00"
