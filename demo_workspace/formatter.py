"""Currency and report formatting for the demo ledger CLI."""


def format_currency(amount: float, symbol: str = "$") -> str:
    """Format a monetary amount with a currency symbol and two decimals."""
    return f"{symbol}{amount:.2f}"


def format_category_line(category: str, total: float, symbol: str = "$") -> str:
    """Format a single category total line for the report."""
    return f"{category}: {format_currency(total, symbol)}"


def format_grand_total(total: float, symbol: str = "$") -> str:
    """Format the grand total line for the report."""
    return f"Grand total: {format_currency(total, symbol)}"


def format_budget_line(budget: float, symbol: str = "$") -> str:
    """Format the budget line for the report."""
    return f"Budget: {format_currency(budget, symbol)}"


def format_remaining_line(remaining: float, symbol: str = "$") -> str:
    """Format the remaining-budget line for the report."""
    return f"Remaining: {format_currency(remaining, symbol)}"


def format_over_budget_line(exceeded: float, symbol: str = "$") -> str:
    """Format the over-budget line for the report."""
    return f"OVER BUDGET by {format_currency(exceeded, symbol)}"


def format_excluded_line(categories: list[str]) -> str:
    """Format the excluded-categories line for the report.

    Categories are expected to already be sorted alphabetically.
    """
    return f"Excluded: {', '.join(categories)}"
