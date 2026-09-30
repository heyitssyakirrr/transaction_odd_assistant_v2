from __future__ import annotations

"""Mechanical monthly comparisons for the LLM and source-grounded display text.

These functions do not assign risk or decide whether a pattern is material.
"""

from decimal import Decimal
from typing import Any


def _amount(row: dict[str, Any], field: str) -> Decimal:
    return Decimal(str(row[field]))


def _money(value: Decimal) -> str:
    return f"{value:,.2f}"


def _count_label(count: int, noun: str) -> str:
    return f"{count} {noun}{'' if count == 1 else 's'}"


def _direction(first: Decimal, second: Decimal) -> str:
    return "rose" if second > first else "fell" if second < first else "held steady"


def _directional_pair(rows: list[dict[str, Any]], field: str, increase: bool) -> str:
    changes = []
    for first, second in zip(rows, rows[1:]):
        difference = _amount(second, field) - _amount(first, field)
        if difference != 0 and (difference > 0) == increase:
            changes.append((first, second))
    if not changes:
        return "none"
    first, second = max(changes, key=lambda pair: abs(_amount(pair[1], field) - _amount(pair[0], field)))
    display = (lambda row: str(row[field])) if field.endswith("count_monthly") else (lambda row: _money(_amount(row, field)))
    return (
        f"{first['year_month']},{second['year_month']}:"
        f"{display(first)}->{display(second)}"
    )


def transaction_comparison_facts(rows: list[dict[str, Any]]) -> str:
    """Repeat the two checks as labelled facts so a small model need not do arithmetic."""
    lines = ["CHECKED MONTH FACTS — copy values from these lines when choosing activity/value or debit/credit months."]
    for row in rows:
        lines.append(
            f"MONTH|{row['year_month']}|transactions={row['txn_count_monthly']}|"
            f"total={_money(_amount(row, 'total_amount'))}|"
            f"debit_count={row['debit_count_monthly']}|credit_count={row['credit_count_monthly']}|"
            f"debits={_money(_amount(row, 'monthly_debit'))}|credits={_money(_amount(row, 'monthly_credit'))}"
        )
    zero_months = [row["year_month"] for row in rows if row["txn_count_monthly"] == 0]
    debit_only = [row["year_month"] for row in rows if row["debit_count_monthly"] > 0 and row["credit_count_monthly"] == 0]
    credit_only = [row["year_month"] for row in rows if row["credit_count_monthly"] > 0 and row["debit_count_monthly"] == 0]
    mixed = [row["year_month"] for row in rows if row["debit_count_monthly"] > 0 and row["credit_count_monthly"] > 0]
    lines.extend((
        "MONTH_STRUCTURE|zero_activity=" + (",".join(zero_months) or "none")
        + "|debit_only=" + (",".join(debit_only) or "none")
        + "|credit_only=" + (",".join(credit_only) or "none")
        + "|mixed_flow=" + (",".join(mixed) or "none"),
        "ACTIVITY_CANDIDATES|count_increase=" + _directional_pair(rows, "txn_count_monthly", True)
        + "|count_decrease=" + _directional_pair(rows, "txn_count_monthly", False)
        + "|total_increase=" + _directional_pair(rows, "total_amount", True)
        + "|total_decrease=" + _directional_pair(rows, "total_amount", False),
        "FLOW_CANDIDATES|debit_increase=" + _directional_pair(rows, "monthly_debit", True)
        + "|debit_decrease=" + _directional_pair(rows, "monthly_debit", False)
        + "|credit_increase=" + _directional_pair(rows, "monthly_credit", True)
        + "|credit_decrease=" + _directional_pair(rows, "monthly_credit", False),
    ))
    return "\n".join(lines)


def activity_pair_context(rows: list[dict[str, Any]], months: list[str]) -> str:
    by_month = {row["year_month"]: row for row in rows}
    first, second = (by_month[month] for month in months)
    first_count, second_count = first["txn_count_monthly"], second["txn_count_monthly"]
    first_total, second_total = _amount(first, "total_amount"), _amount(second, "total_amount")
    return (
        f"{months[0]} to {months[1]}: transactions {_direction(Decimal(first_count), Decimal(second_count))} "
        f"{first_count} to {second_count}; monthly total {_direction(first_total, second_total)} "
        f"{_money(first_total)} to {_money(second_total)}."
    )


def flow_pair_context(rows: list[dict[str, Any]], months: list[str]) -> str:
    by_month = {row["year_month"]: row for row in rows}
    first, second = (by_month[month] for month in months)
    return (
        f"{months[0]}: {_count_label(first['debit_count_monthly'], 'debit')}/{_money(_amount(first, 'monthly_debit'))}, "
        f"{_count_label(first['credit_count_monthly'], 'credit')}/{_money(_amount(first, 'monthly_credit'))}; "
        f"{months[1]}: {_count_label(second['debit_count_monthly'], 'debit')}/{_money(_amount(second, 'monthly_debit'))}, "
        f"{_count_label(second['credit_count_monthly'], 'credit')}/{_money(_amount(second, 'monthly_credit'))}."
    )


def activity_six_month_context(rows: list[dict[str, Any]]) -> str:
    counts = [row["txn_count_monthly"] for row in rows]
    totals = [_amount(row, "total_amount") for row in rows]
    return (
        f"Six-month transaction counts range {min(counts)}-{max(counts)}; "
        f"monthly totals range {_money(min(totals))}-{_money(max(totals))}."
    )


def flow_six_month_context(rows: list[dict[str, Any]]) -> str:
    debit_count = sum(row["debit_count_monthly"] for row in rows)
    credit_count = sum(row["credit_count_monthly"] for row in rows)
    debits = sum((_amount(row, "monthly_debit") for row in rows), Decimal(0))
    credits = sum((_amount(row, "monthly_credit") for row in rows), Decimal(0))
    return (
        f"Six-month flow: {_count_label(debit_count, 'debit')} totalling {_money(debits)}; "
        f"{_count_label(credit_count, 'credit')} totalling {_money(credits)}."
    )
