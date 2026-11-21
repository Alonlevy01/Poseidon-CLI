"""Category sorting and budget calculations for the demo ledger CLI."""

from typing import Optional


def sort_categories(totals: dict[str, float], sort_by: str = "name") -> list[str]:
    """Return category names ordered by the requested sort strategy.

    "name" sorts alphabetically; "amount" sorts by total descending.
    """
    if sort_by == "amount":
        return sorted(totals, key=lambda c: totals[c], reverse=True)
    return sorted(totals)


def calculate_grand_total(totals: dict[str, float]) -> float:
    """Return the sum of all category totals."""
    return sum(totals.values())


def calculate_budget_status(
    grand_total: float, budget: Optional[float]
) -> dict[str, object]:
    """Compare the grand total against an optional budget.

    Returns a dict with keys:
        "budget"        - the budget value or None
        "remaining"     - budget - grand_total (None if no budget)
        "exceeded"      - grand_total - budget when over (None if no budget)
        "over_budget"   - True when a budget is set and grand_total exceeds it
    """
    if budget is None:
        return {
            "budget": None,
            "remaining": None,
            "exceeded": None,
            "over_budget": False,
        }

    remaining = budget - grand_total
    over_budget = remaining < 0
    exceeded = grand_total - budget if over_budget else None

    return {
        "budget": budget,
        "remaining": remaining,
        "exceeded": exceeded,
        "over_budget": over_budget,
    }
