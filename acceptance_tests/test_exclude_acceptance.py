"""Hidden Milestone 7 acceptance test for the Milestone 5 --exclude multi-file request.

Kept strictly outside demo_workspace/, prompts, and search indexes until both model runs finish.
Tests the delivered program on an unseen input combining:
- multiple excluded categories with extra whitespace ("gym, travel")
- custom currency prefix ("GBP ")
- budget recalculation against the *filtered* total (not unfiltered total)
- alphabetical ordering of the Excluded summary line
"""

import subprocess
import sys
from pathlib import Path

LEDGER_SCRIPT = Path(__file__).parent / "ledger.py"


def test_exclude_unseen_input_filters_before_budget_and_formats_report() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            str(LEDGER_SCRIPT),
            "--expenses",
            "travel:150,coffee:12.50,books:45,coffee:7.50,gym:60",
            "--exclude",
            "gym, travel",
            "--budget",
            "60",
            "--currency",
            "GBP ",
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )
    assert proc.returncode == 0, f"Program failed to run: {proc.stderr}"
    output = proc.stdout.strip()
    lines = [line.strip() for line in output.splitlines() if line.strip()]

    # Excluded categories must not appear as expense lines
    expense_lines = [line for line in lines if not line.startswith("Excluded:")]
    assert not any(line.startswith("gym:") for line in expense_lines), output
    assert not any(line.startswith("travel:") for line in expense_lines), output

    # Remaining categories and filtered grand total must be exact
    assert "books: GBP 45.00" in lines, output
    assert "coffee: GBP 20.00" in lines, output
    assert "Grand total: GBP 65.00" in lines, output

    # Budget must compare against filtered total (65.00 vs 60.00 -> over by 5.00, NOT 215.00)
    assert "Budget: GBP 60.00" in lines, output
    assert "OVER BUDGET by GBP 5.00" in lines, output

    # Excluded summary line must list excluded categories sorted alphabetically
    assert "Excluded: gym, travel" in lines, output
