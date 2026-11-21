"""CLI tool to sum expenses by category from a comma-separated list of category:amount pairs."""

import argparse

from analytics import calculate_budget_status, calculate_grand_total, filter_excluded_categories, sort_categories
from formatter import (
    format_budget_line,
    format_excluded_line,
    format_category_line,
    format_grand_total,
    format_over_budget_line,
    format_remaining_line,
)
from parser import parse_expenses

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
    parser.add_argument(
        "--exclude",
        default="",
        help="Comma-separated list of categories to ignore before computing totals.",
    )
    parser.add_argument(
        "--sort",
        choices=["name", "amount"],
        default="name",
        help="Sort categories by name (alphabetical) or amount (highest total first).",
    )

    args = parser.parse_args()

    try:
        totals = parse_expenses(args.expenses)
    except argparse.ArgumentTypeError as exc:
        parser.error(str(exc))

    if not totals:
        print("No expenses provided.")
        return

    excluded = [cat.strip() for cat in args.exclude.split(",") if cat.strip()]
    excluded_sorted = sorted(excluded)
    if excluded_sorted:
        totals = filter_excluded_categories(totals, excluded_sorted)

    if not totals:
        print("No expenses provided.")
        return

    ordered = sort_categories(totals, args.sort)
    for category in ordered:
        category_total = totals[category]
        print(format_category_line(category, category_total, args.currency))

    grand_total = calculate_grand_total(totals)
    print(format_grand_total(grand_total, args.currency))

    if excluded_sorted:
        print(format_excluded_line(excluded_sorted))

    if args.budget is not None:
        status = calculate_budget_status(grand_total, args.budget)
        print(format_budget_line(status["budget"], args.currency))
        if status["over_budget"]:
            print(format_over_budget_line(status["exceeded"], args.currency))
        else:
            print(format_remaining_line(status["remaining"], args.currency))


if __name__ == "__main__":
    main()
