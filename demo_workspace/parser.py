"""Expense parsing and validation for the demo ledger CLI."""

import argparse


def parse_expenses(value: str) -> dict[str, float]:
    """Parse a comma-separated string of category:amount pairs into a dict of category totals.

    Validates that each amount is a positive number. Raises argparse.ArgumentTypeError
    on any malformed or invalid input so argparse can display a clean error message.
    """
    if not value:
        return {}

    totals: dict[str, float] = {}
    pairs = value.split(",")

    for raw_pair in pairs:
        pair = raw_pair.strip()
        if not pair:
            continue

        if ":" not in pair:
            raise argparse.ArgumentTypeError(
                f"Invalid expense pair '{pair}': expected format category:amount"
            )

        category, amount_str = pair.split(":", 1)
        category = category.strip()
        amount_str = amount_str.strip()

        if not category:
            raise argparse.ArgumentTypeError(
                f"Invalid expense pair '{pair}': category name is empty"
            )

        try:
            amount = float(amount_str)
        except ValueError:
            raise argparse.ArgumentTypeError(
                f"Invalid amount '{amount_str}' in pair '{pair}': not a number"
            )

        if amount <= 0:
            raise argparse.ArgumentTypeError(
                f"Invalid amount {amount} in pair '{pair}': must be a positive number"
            )

        totals[category] = totals.get(category, 0.0) + amount

    return totals
