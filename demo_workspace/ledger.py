"""CLI tool to sum expenses by category from a comma-separated list of category:amount pairs."""

import argparse


def parse_expenses(value: str) -> dict:
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Group and total expenses by category."
    )
    parser.add_argument(
        "--expenses",
        required=True,
        help='Comma-separated category:amount pairs, e.g. "food:12.50,rent:800,food:7.50"',
    )
    parser.add_argument(
        "--budget",
        type=float,
        default=None,
        help="Optional budget limit to compare against the grand total.",
    )
    parser.add_argument(
        "--currency",
        default="$",
        help="Currency symbol to prefix monetary amounts with.",
    )

    args = parser.parse_args()

    try:
        totals = parse_expenses(args.expenses)
    except argparse.ArgumentTypeError as exc:
        parser.error(str(exc))

    if not totals:
        print("No expenses provided.")
        return

    grand_total = 0.0
    for category in sorted(totals):
        category_total = totals[category]
        grand_total += category_total
        print(f"{category}: {args.currency}{category_total:.2f}")

    print(f"Grand total: {args.currency}{grand_total:.2f}")

    if args.budget is not None:
        print(f"Budget: {args.currency}{args.budget:.2f}")
        remaining = args.budget - grand_total
        if remaining >= 0:
            print(f"Remaining: {args.currency}{remaining:.2f}")
        else:
            exceeded = grand_total - args.budget
            print(f"OVER BUDGET by {args.currency}{exceeded:.2f}")


if __name__ == "__main__":
    main()
