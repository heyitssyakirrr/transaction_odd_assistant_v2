from __future__ import annotations

"""Mechanical monthly comparisons supplied as facts to the LLM.

This module does not write review insights, assign risk, or decide materiality.
"""

from decimal import Decimal
from typing import Any


def _amount(row: dict[str, Any], field: str) -> Decimal:
    return Decimal(str(row[field]))


def _money(value: Decimal) -> str:
    return f"{value:,.2f}"


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
    """Label source values and mechanical comparisons for the transaction prompt."""
    lines = ["CHECKED MONTH FACTS — use these values to choose activity/value or debit/credit evidence months."]
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
