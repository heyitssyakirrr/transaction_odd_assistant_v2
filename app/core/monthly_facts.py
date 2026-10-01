from __future__ import annotations

"""Mechanical monthly comparisons for the LLM and source-grounded display text.

These functions do not assign risk or decide whether a pattern is material.
"""

from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Literal

# ReviewCheck.rationale allows 360 characters; stay inside it.
_RATIONALE_LIMIT = 360


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


# --- Activity after an inactive period -------------------------------------
# Whether this pattern exists is arithmetic on the six rows, so the code decides
# it. The model still judges the overall risk. Wording stays observational: six
# rows cannot establish formal dormancy or that the account was ever active.

Direction = Literal["debit", "credit", "mixed", "unspecified"]
AmountBand = Literal["above", "below"]


@dataclass(frozen=True)
class InactivityRun:
    """Consecutive zero-transaction months followed by the first active month."""

    zero_start: str
    zero_end: str
    zero_months: int
    active_month: str
    transactions: int
    debit_count: int
    credit_count: int
    total: Decimal
    debit: Decimal
    credit: Decimal
    largest_single: Decimal
    direction: Direction
    amount_band: AmountBand
    # Which reference was reached, and its value; None when below reference.
    trigger: Literal["single", "total"] | None
    trigger_reference: Decimal | None
    later_active_months: int
    later_transactions: int
    later_total: Decimal


def find_inactivity_run(
    rows: list[dict[str, Any]], *, min_zero_months: int,
    single_reference: Decimal, month_total_reference: Decimal,
) -> InactivityRun | None:
    """Return the most recent run of ``min_zero_months`` or more zero-count months
    immediately followed by a month with transactions, or None.

    Rows are the six consecutive months in order (guaranteed by the CSV loader).
    The amount band uses the largest single transaction and the monthly total, so
    one large payment and many mid-sized payments are both recognised.
    """
    for index in range(len(rows) - 1, 0, -1):
        if rows[index]["txn_count_monthly"] <= 0:
            continue
        start = index
        while start > 0 and rows[start - 1]["txn_count_monthly"] == 0:
            start -= 1
        zeros = index - start
        if zeros < min_zero_months:
            continue
        active = rows[index]
        total, largest = _amount(active, "total_amount"), _amount(active, "max_amount")
        debit, credit = _amount(active, "monthly_debit"), _amount(active, "monthly_credit")
        has_debit = active["debit_count_monthly"] > 0 or debit > 0
        has_credit = active["credit_count_monthly"] > 0 or credit > 0
        direction: Direction = (
            "mixed" if has_debit and has_credit else "debit" if has_debit
            else "credit" if has_credit else "unspecified"
        )
        trigger: Literal["single", "total"] | None = None
        reference: Decimal | None = None
        if largest >= single_reference:
            trigger, reference = "single", single_reference
        elif total >= month_total_reference:
            trigger, reference = "total", month_total_reference
        later = [row for row in rows[index + 1:] if row["txn_count_monthly"] > 0]
        return InactivityRun(
            zero_start=rows[start]["year_month"], zero_end=rows[index - 1]["year_month"], zero_months=zeros,
            active_month=active["year_month"], transactions=active["txn_count_monthly"],
            debit_count=active["debit_count_monthly"], credit_count=active["credit_count_monthly"],
            total=total, debit=debit, credit=credit, largest_single=largest, direction=direction,
            amount_band="above" if trigger else "below", trigger=trigger, trigger_reference=reference,
            later_active_months=len(later), later_transactions=sum(row["txn_count_monthly"] for row in later),
            later_total=sum((_amount(row, "total_amount") for row in later), Decimal(0)),
        )
    return None


def inactivity_run_fact(run: InactivityRun | None, min_zero_months: int) -> str:
    """One labelled prompt line. It states facts and a band, never a risk level."""
    if run is None:
        return f"INACTIVITY_RUN|status=none|rule={min_zero_months}_or_more_zero_months_then_activity"
    return (
        f"INACTIVITY_RUN|status=present|zero_start={run.zero_start}|zero_end={run.zero_end}|"
        f"zero_months={run.zero_months}|active_month={run.active_month}|transactions={run.transactions}|"
        f"debits={run.debit_count}/{_money(run.debit)}|credits={run.credit_count}/{_money(run.credit)}|"
        f"largest_single={_money(run.largest_single)}|direction={run.direction}|"
        f"amount_vs_reference={run.amount_band}|later_active_months={run.later_active_months}"
    )


def _flow_phrase(run: InactivityRun) -> str:
    debits = f"{_count_label(run.debit_count, 'debit')} of {_money(run.debit)}"
    credits = f"{_count_label(run.credit_count, 'credit')} of {_money(run.credit)}"
    if run.direction == "debit":
        return debits
    if run.direction == "credit":
        return credits
    if run.direction == "mixed":
        return f"{debits} and {credits}"
    return _count_label(run.transactions, "transaction")


def _staff_check(run: InactivityRun) -> str:
    if run.amount_band == "below":
        return "Amount is below the review reference: looks like renewed use with limited movement."
    scope = "Largest single transaction" if run.trigger == "single" else "Monthly total"
    reached = f"{scope} is at or above the {_money(run.trigger_reference or Decimal(0))} review reference."
    ask = {
        "debit": "Suggested check: confirm the payment purpose and how the account was funded.",
        "credit": "Suggested check: confirm the source and purpose of the incoming funds.",
    }.get(run.direction, "Suggested check: confirm the payment purpose and the source of incoming funds.")
    return f"{reached} {ask}"


def dormancy_rationale(run: InactivityRun) -> str:
    """Deterministic staff text: the pattern, the amount in context, and what to check.

    Every figure comes from the rows. No source of funds is asserted, and the
    result is never called formal dormancy. Optional sentences are dropped, in
    order, if the text would exceed the rationale limit.
    """
    pattern = (
        f"No transactions for {run.zero_months} months ({run.zero_start} to {run.zero_end}); "
        f"activity resumed in {run.active_month} with {_flow_phrase(run)}"
        + (f" (largest single {_money(run.largest_single)})." if run.transactions > 1 else ".")
    )
    since = (
        f"Since then: {_count_label(run.later_active_months, 'active month')}, "
        f"{run.later_transactions} transactions, total {_money(run.later_total)}."
        if run.later_active_months else ""
    )
    caveat = "Six-month view only; not a formal dormancy status."
    for parts in ((pattern, since, _staff_check(run), caveat), (pattern, _staff_check(run), caveat),
                  (pattern, _staff_check(run)), (pattern,)):
        text = " ".join(part for part in parts if part)
        if len(text) <= _RATIONALE_LIMIT:
            return text
    return pattern[:_RATIONALE_LIMIT]


def no_inactivity_rationale(min_zero_months: int) -> str:
    return (
        f"No run of {min_zero_months} or more consecutive zero-transaction months "
        "followed by activity in this six-month review."
    )