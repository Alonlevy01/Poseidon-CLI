"""Held-out second acceptance test for the Exercise 1 Optional Challenge benchmark.

Kept strictly outside demo_workspace/ and development prompts. Verifies a brand-new
multi-file feature (--min-amount threshold filtering) on an unseen input.
"""

import subprocess
import sys
from pathlib import Path

LEDGER_SCRIPT = Path(__file__).parent / "ledger.py"


def test_min_amount_filters_below_threshold_before_budget() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            str(LEDGER_SCRIPT),
            "--expenses",
            "rent:800,coffee:4.50,snacks:8.00,groceries:120,coffee:3.50",
            "--min-amount",
            "10",
            "--budget",
            "950",
            "--currency",
            "EUR ",
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )
    assert proc.returncode == 0, f"Program failed to run: {proc.stderr}"
    output = proc.stdout.strip()
    lines = [line.strip() for line in output.splitlines() if line.strip()]

    # coffee (8.00) and snacks (8.00) are strictly below 10.00 and must be filtered out
    expense_lines = [line for line in lines if not line.lower().startswith("filtered")]
    assert not any(line.startswith("coffee:") for line in expense_lines), output
    assert not any(line.startswith("snacks:") for line in expense_lines), output

    # Remaining totals: groceries=120.00, rent=800.00 -> Grand total=920.00, Remaining=30.00
    assert "groceries: EUR 120.00" in lines, output
    assert "rent: EUR 800.00" in lines, output
    assert "Grand total: EUR 920.00" in lines, output
    assert "Budget: EUR 950.00" in lines, output
    assert "Remaining: EUR 30.00" in lines, output
    assert any("coffee" in line and "snacks" in line for line in lines), output
