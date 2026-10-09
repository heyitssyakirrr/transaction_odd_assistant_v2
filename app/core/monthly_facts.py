from __future__ import annotations

"""Mechanical monthly comparisons for the LLM and source-grounded display text.

These functions do not assign risk or decide whether a pattern is material.
"""

import re
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Literal

# ReviewCheck.rationale allows 360 characters; stay inside it.
_RATIONALE_LIMIT = 360


def _amount(row: dict[str, Any], field: str) -> Decimal:
    return Decimal(str(row[field]))


def _money(value: Decimal) -> str:
    return f"{value:,.2f}"


def _rm(value: Decimal) -> str:
    """Staff-facing amount, e.g. RM 10,000.00."""
    return f"RM {_money(value)}"


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


def month_fact_line(row: dict[str, Any]) -> str:
    """One month as labelled fields, shared by call 1 and the summary.

    Labelled ``key=value`` fields, not bare columns: Qwen 7B miscounts unlabelled
    ``|`` columns (logged 2026-10-05: a month total was reported as credits).
    """
    return (
        f"MONTH|{row['year_month']}|transactions={row['txn_count_monthly']}|"
        f"total={_money(_amount(row, 'total_amount'))}|"
        f"debit_count={row['debit_count_monthly']}|credit_count={row['credit_count_monthly']}|"
        f"debits={_money(_amount(row, 'monthly_debit'))}|credits={_money(_amount(row, 'monthly_credit'))}"
    )


def transaction_comparison_facts(rows: list[dict[str, Any]]) -> str:
    """Repeat the two checks as labelled facts so a small model need not do arithmetic."""
    lines = ["CHECKED MONTH FACTS — copy values from these lines when choosing activity/value or debit/credit months."]
    lines.extend(month_fact_line(row) for row in rows)
    zero_months = [row["year_month"] for row in rows if row["txn_count_monthly"] == 0]
    debit_only = [row["year_month"] for row in rows if row["debit_count_monthly"] > 0 and row["credit_count_monthly"] == 0]
    credit_only = [row["year_month"] for row in rows if row["credit_count_monthly"] > 0 and row["debit_count_monthly"] == 0]
    mixed = [row["year_month"] for row in rows if row["debit_count_monthly"] > 0 and row["credit_count_monthly"] > 0]
    lines.extend((
        "MONTH_STRUCTURE|zero_activity=" + (",".join(zero_months) or "none")
        + "|debit_only=" + (",".join(debit_only) or "none")
        + "|credit_only=" + (",".join(credit_only) or "none")
        + "|mixed_flow=" + (",".join(mixed) or "none"),
        "FLOW_CANDIDATES|debit_increase=" + _directional_pair(rows, "monthly_debit", True)
        + "|debit_decrease=" + _directional_pair(rows, "monthly_debit", False)
        + "|credit_increase=" + _directional_pair(rows, "monthly_credit", True)
        + "|credit_decrease=" + _directional_pair(rows, "monthly_credit", False),
    ))
    return "\n".join(lines)


def _amount_phrase(row: dict[str, Any]) -> str:
    return (
        f"{_count_label(row['txn_count_monthly'], 'transaction')}, total {_rm(_amount(row, 'total_amount'))}, "
        f"largest single {_rm(_amount(row, 'max_amount'))}"
    )


def _share(part: Decimal, whole: Decimal) -> str:
    return f"{(part / whole * 100).quantize(Decimal('0.1'))}%" if whole else "0.0%"


def amount_facts(rows: list[dict[str, Any]], single_reference: Decimal) -> str:
    """Labelled amount facts per month and the main amount candidates, for call 1.

    largest_single is the largest single transaction in the month; largest_share is
    that transaction as a share of the month's total. The model decides what matters.
    """
    lines = ["AMOUNT_FACTS — amounts in RM; average and std are the upstream avg_amount and std_amount."]
    for row in rows:
        lines.append(
            f"AMOUNT|{row['year_month']}|transactions={row['txn_count_monthly']}|"
            f"total={_money(_amount(row, 'total_amount'))}|average={_money(_amount(row, 'avg_amount'))}|"
            f"std={_money(_amount(row, 'std_amount'))}|largest_single={_money(_amount(row, 'max_amount'))}|"
            f"largest_share_of_total={_share(_amount(row, 'max_amount'), _amount(row, 'total_amount'))}"
        )
    largest = max(rows, key=lambda row: _amount(row, "max_amount"))
    reference = "above" if _amount(largest, "max_amount") >= single_reference else "below"
    lines.append(
        f"AMOUNT_CANDIDATES|largest_single_in_six_months={largest['year_month']}:{_money(_amount(largest, 'max_amount'))}|"
        f"largest_single_vs_review_reference={reference} (reference {_money(single_reference)})"
    )
    return "\n".join(lines)


def activity_pair_context(rows: list[dict[str, Any]], months: list[str]) -> str:
    """Key figures for one or two selected months, led by the amounts."""
    by_month = {row["year_month"]: row for row in rows}
    if len(months) == 1:
        return f"{months[0]}: {_amount_phrase(by_month[months[0]])}."
    first, second = (by_month[month] for month in months)
    return (
        f"{months[0]} to {months[1]}: total {_rm(_amount(first, 'total_amount'))} to {_rm(_amount(second, 'total_amount'))}; "
        f"largest single {_rm(_amount(first, 'max_amount'))} to {_rm(_amount(second, 'max_amount'))}; "
        f"transactions {first['txn_count_monthly']} to {second['txn_count_monthly']}."
    )


def flow_pair_context(rows: list[dict[str, Any]], months: list[str]) -> str:
    """Debits and credits of the selected months; one month is shown after the month before it."""
    index = {row["year_month"]: position for position, row in enumerate(rows)}
    shown = [rows[index[months[0]] - 1]["year_month"], *months] if len(months) == 1 and index[months[0]] > 0 else months
    return "; ".join(
        f"{row['year_month']}: {_count_label(row['debit_count_monthly'], 'debit')} {_rm(_amount(row, 'monthly_debit'))}, "
        f"{_count_label(row['credit_count_monthly'], 'credit')} {_rm(_amount(row, 'monthly_credit'))}"
        for row in (rows[index[month]] for month in shown)
    ) + "."


def activity_six_month_context(rows: list[dict[str, Any]]) -> str:
    totals = [_amount(row, "total_amount") for row in rows]
    largest = max(rows, key=lambda row: _amount(row, "max_amount"))
    return (
        f"Six-month totals range {_rm(min(totals))} to {_rm(max(totals))}; largest single transaction "
        f"{_rm(_amount(largest, 'max_amount'))} in {largest['year_month']}."
    )


def flow_six_month_context(rows: list[dict[str, Any]]) -> str:
    debit_count = sum(row["debit_count_monthly"] for row in rows)
    credit_count = sum(row["credit_count_monthly"] for row in rows)
    debits = sum((_amount(row, "monthly_debit") for row in rows), Decimal(0))
    credits = sum((_amount(row, "monthly_credit") for row in rows), Decimal(0))
    return (
        f"Six-month flow: {_count_label(debit_count, 'debit')} totalling {_rm(debits)}; "
        f"{_count_label(credit_count, 'credit')} totalling {_rm(credits)}."
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
    debits = f"{_count_label(run.debit_count, 'debit')} of {_rm(run.debit)}"
    credits = f"{_count_label(run.credit_count, 'credit')} of {_rm(run.credit)}"
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
    reached = f"{scope} is at or above the {_rm(run.trigger_reference or Decimal(0))} review reference."
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
        + (f" (largest single {_rm(run.largest_single)})." if run.transactions > 1 else ".")
    )
    since = (
        f"Since then: {_count_label(run.later_active_months, 'active month')}, "
        f"{run.later_transactions} transactions, total {_rm(run.later_total)}."
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


# --- Burst and transaction-gap timing ---------------------------------------
# Upstream definitions (confirmed by the data owner):
# - pct_burst: share of the month's activity in which the same counterparty
#   transacted more than 3 times that month ("burst"). Shown as a percentage.
# - pct_trx_gap: average gap in days between transactions in the month. A gap
#   needs two transactions, so in a one-transaction month it measures back to
#   the previous transaction, which may fall before the six-month window.
# The model judges what the timing means; these helpers only lay the values out.

GapBasis = Literal["in_month", "since_previous", "no_activity"]
# A month is a burst month when its burst share is this or more (any number of transactions).
# The prompt, the pattern check, the "says no burst" check and RISK_FACTS all use it.
BURST_GUIDE_SHARE = Decimal("25.0")


def _burst_share(row: dict[str, Any]) -> Decimal:
    return (Decimal(str(row["pct_burst"])) * 100).quantize(Decimal("0.1"))


def _gap_days(row: dict[str, Any]) -> Decimal:
    return Decimal(str(row["pct_trx_gap"])).quantize(Decimal("0.01"))


def _days(value: Decimal) -> str:
    """Staff-facing day count: 364 rather than 364.00, 10.67 stays 10.67."""
    return f"{value.normalize():f}"


def gap_basis(row: dict[str, Any]) -> GapBasis:
    count = row["txn_count_monthly"]
    if count >= 2:
        return "in_month"
    return "since_previous" if count == 1 and _gap_days(row) > 0 else "no_activity"


def has_in_month_gap(rows: list[dict[str, Any]]) -> bool:
    return any(gap_basis(row) == "in_month" for row in rows)


def burst_guide_months(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The burst months: burst share of 25% or more."""
    return [row for row in rows if _burst_share(row) >= BURST_GUIDE_SHARE]


def risk_facts(rows: list[dict[str, Any]], single_reference: Decimal) -> str:
    """The two facts the summary's low-risk rule reads: amount size and notable bursts."""
    largest = max(rows, key=lambda row: _amount(row, "max_amount"))
    below = "yes" if _amount(largest, "max_amount") < single_reference else "no"
    guide = ",".join(f"{row['year_month']}:{_burst_share(row)}%" for row in burst_guide_months(rows)) or "none"
    return (
        f"RISK_FACTS|largest_single=RM {_money(_amount(largest, 'max_amount'))} in {largest['year_month']}|"
        f"review_reference=RM {_money(single_reference)}|largest_single_below_reference={below}|burst_guide_months={guide}"
    )


def long_gap_months(rows: list[dict[str, Any]], long_gap_days: Decimal) -> list[dict[str, Any]]:
    """One-transaction months whose gap back to the previous transaction is ``long_gap_days`` or more."""
    return [row for row in rows if gap_basis(row) == "since_previous" and _gap_days(row) >= long_gap_days]


def burst_gap_facts(rows: list[dict[str, Any]], long_gap_days: Decimal) -> str:
    """Labelled per-month timing facts and a few candidates, for the timing prompt."""
    lines = [
        "BURST_GAP_FACTS — burst_share = share of the month's activity where the same counterparty transacted more "
        "than 3 times that month. avg_gap_days = average days between transactions; when gap_basis=since_previous "
        "the month has one transaction and the gap reaches back to the previous transaction.",
    ]
    for row in rows:
        lines.append(
            f"BURST|{row['year_month']}|transactions={row['txn_count_monthly']}|"
            f"burst_share={_burst_share(row)}%|avg_gap_days={_gap_days(row)}|gap_basis={gap_basis(row)}"
        )
    lines.append(
        "BURST_GUIDE|burst_months_with_burst_share_25%_or_more="
        + (",".join(f"{row['year_month']}:{_burst_share(row)}%" for row in burst_guide_months(rows)) or "no month")
    )
    days = _days(long_gap_days)
    lines.append(
        f"GAP_GUIDE|months_with_{days}_or_more_days_since_the_previous_transaction="
        + (",".join(f"{row['year_month']}:{_days(_gap_days(row))} days" for row in long_gap_months(rows, long_gap_days))
           or "no month")
    )
    return "\n".join(lines)


def burst_gap_note(rows: list[dict[str, Any]], long_gap_days: Decimal) -> str:
    """The burst and long-gap findings for this account, in plain words, for the end of the call 2 prompt.

    Qwen 7B read five 0.0% rows as "no burst" and an average gap of 7.33 days as a long
    gap, although BURST_GUIDE and GAP_GUIDE said otherwise (logged 2026-10-06). Stating
    the two findings at the end, as the inactivity note does, keeps it on the facts.
    """
    burst = burst_guide_months(rows)
    gaps = long_gap_months(rows, long_gap_days)
    days = _days(long_gap_days)
    burst_text = (
        "This account HAS burst months (burst share 25.0% or more): "
        + ", ".join(f"{row['year_month']} ({_burst_share(row)}%)" for row in burst) + "."
        if burst else "This account has NO burst month: no month reaches a 25.0% burst share."
    )
    gap_text = (
        f"Long gaps ({days} days or more since the previous transaction): "
        + ", ".join(f"{row['year_month']} ({_days(_gap_days(row))} days)" for row in gaps) + "."
        if gaps else f"No long gap: no transaction came {days} or more days after the previous one, so do not call "
                     "any gap long."
    )
    outcome = "pattern_found" if burst or gaps else "no_pattern_found"
    return (
        f"\nBURST AND GAP NOTE: {burst_text} {gap_text} burst_gaps_insight must state this burst finding and this "
        f"gap finding. burst_gaps_outcome is {outcome}.\n"
    )


def _burst_month_phrase(row: dict[str, Any]) -> str:
    basis = gap_basis(row)
    gap = (
        f"avg gap {_days(_gap_days(row))} days" if basis == "in_month"
        else f"{_days(_gap_days(row))} days since the previous transaction" if basis == "since_previous"
        else "no gap figure"
    )
    return f"{row['year_month']}: {_count_label(row['txn_count_monthly'], 'transaction')}, burst {_burst_share(row)}%, {gap}"


def burst_gap_pair_facts(rows: list[dict[str, Any]], months: list[str]) -> str:
    """Exact values for the given months, as a short factual line."""
    by_month = {row["year_month"]: row for row in rows}
    return "; ".join(_burst_month_phrase(by_month[month]) for month in months) + "."


def burst_gap_evidence_months(rows: list[dict[str, Any]]) -> list[str]:
    """Up to two months that best show the timing, for evidence when nothing is selected.

    The peak burst month first, then the longest gap (in-month, else back to the
    previous transaction). Chronological order.
    """
    months: list[str] = []
    burst = burst_guide_months(rows)
    if burst:
        months.append(max(burst, key=_burst_share)["year_month"])
    in_month = [row for row in rows if gap_basis(row) == "in_month"]
    gap_rows = in_month or [row for row in rows if gap_basis(row) == "since_previous"]
    if gap_rows:
        candidate = max(gap_rows, key=_gap_days)["year_month"]
        if candidate not in months:
            months.append(candidate)
    return sorted(months)


def burst_gap_six_month_context(rows: list[dict[str, Any]]) -> str:
    burst = burst_guide_months(rows)
    in_month = [row for row in rows if gap_basis(row) == "in_month"]
    since_previous = [row for row in rows if gap_basis(row) == "since_previous"]
    peak = max(rows, key=_burst_share)
    if burst:
        burst_text = "Burst (25% or more of a month's activity with one counterparty) in " + ", ".join(
            f"{row['year_month']} ({_burst_share(row)}%)" for row in burst)
    else:
        burst_text = (f"No month reaches the 25% burst level (highest {_burst_share(peak)}% in {peak['year_month']})"
                      if _burst_share(peak) > 0 else "No burst activity in any month")
    if in_month:
        gaps = [_gap_days(row) for row in in_month]
        gap_text = f"average gap {_days(min(gaps))}-{_days(max(gaps))} days in months with 2+ transactions"
    elif since_previous:
        back = max(since_previous, key=_gap_days)
        gap_text = f"the {back['year_month']} transaction came {_days(_gap_days(back))} days after the previous one"
    else:
        gap_text = "no gap figure is available"
    return f"{burst_text}; {gap_text}."


def allowed_burst_gap_numbers(rows: list[dict[str, Any]]) -> set[Decimal]:
    """Every number the timing insight may quote: months, counts, shares and gaps."""
    allowed: set[Decimal] = set(Decimal(n) for n in range(0, 13))
    for row in rows:
        month = row["year_month"]
        allowed.update({Decimal(month), Decimal(month[:4]), Decimal(row["txn_count_monthly"])})
        share, gap = Decimal(str(row["pct_burst"])) * 100, Decimal(str(row["pct_trx_gap"]))
        for value in (share, gap, Decimal(str(row["pct_burst"]))):
            for places in ("1", "0.1", "0.01"):
                allowed.add(value.quantize(Decimal(places)).normalize())
    return {value.normalize() for value in allowed}


# --- Evidence tables ----------------------------------------------------------
# Every check shows all six months so staff see the context; the months the model
# selected are highlighted. Values are formatted once here, so the page and the
# HTML report always show identical figures.

def _count(row: dict[str, Any], field: str) -> str:
    return str(row[field])


def _avg_days_cell(row: dict[str, Any]) -> str:
    basis = gap_basis(row)
    if basis == "in_month":
        return _days(_gap_days(row))
    if basis == "since_previous":
        return f"{_days(_gap_days(row))} (since previous transaction)"
    return "-"


_AMOUNT_COLUMNS: tuple[tuple[str, Any], ...] = (
    ("Month", lambda r: r["year_month"]),
    ("No. of transactions", lambda r: _count(r, "txn_count_monthly")),
    ("Total amount (RM)", lambda r: _money(_amount(r, "total_amount"))),
    ("Average amount (RM)", lambda r: _money(_amount(r, "avg_amount"))),
    ("Spread of amounts, std dev (RM)", lambda r: _money(_amount(r, "std_amount"))),
    ("Largest single transaction (RM)", lambda r: _money(_amount(r, "max_amount"))),
)

_TABLE_COLUMNS: dict[str, tuple[tuple[str, Any], ...]] = {
    "activity_after_inactivity": _AMOUNT_COLUMNS,
    "activity_and_amount_change": _AMOUNT_COLUMNS,
    "money_in_and_out": (
        ("Month", lambda r: r["year_month"]),
        ("No. of debits (money out)", lambda r: _count(r, "debit_count_monthly")),
        ("Debit amount (RM)", lambda r: _money(_amount(r, "monthly_debit"))),
        ("No. of credits (money in)", lambda r: _count(r, "credit_count_monthly")),
        ("Credit amount (RM)", lambda r: _money(_amount(r, "monthly_credit"))),
    ),
    "burst_and_gaps": (
        ("Month", lambda r: r["year_month"]),
        ("No. of transactions", lambda r: _count(r, "txn_count_monthly")),
        ("Burst % (same counterparty more than 3 times in the month)", lambda r: f"{_burst_share(r)}%"),
        ("Average days between transactions", _avg_days_cell),
    ),
}


def check_table(check: str, rows: list[dict[str, Any]], highlight_months: list[str]) -> tuple[list[str], list[list[str]], list[int]]:
    """(columns, rows, highlighted row indexes) for one check's evidence table."""
    spec = _TABLE_COLUMNS[check]
    body = [[render(row) for _, render in spec] for row in rows]
    highlighted = [index for index, row in enumerate(rows) if row["year_month"] in highlight_months]
    return [label for label, _ in spec], body, highlighted


# --- Debit/credit type of each month -----------------------------------------------

def _structure(row: dict[str, Any]) -> str:
    debit, credit = row["debit_count_monthly"] > 0, row["credit_count_monthly"] > 0
    return "mixed" if debit and credit else "debit_only" if debit else "credit_only" if credit else "zero"


_STRUCTURE_WORDS = {
    "mixed": "debits and credits", "debit_only": "debits only", "credit_only": "credits only", "zero": "no transactions",
}


def debit_credit_mix_facts(rows: list[dict[str, Any]]) -> str:
    """Each month's debit/credit type in order, and every change between neighbouring months.

    Qwen 7B misreads the grouped MONTH_STRUCTURE line; the same facts laid out month by
    month, with the changes already listed, let it pick a correct pair. It still decides.
    """
    changes = [
        f"{before['year_month']} {_structure(before)} -> {after['year_month']} {_structure(after)}"
        for before, after in zip(rows, rows[1:]) if _structure(before) != _structure(after)
    ]
    return "\n".join((
        "DEBIT_CREDIT_BY_MONTH|" + "|".join(f"{row['year_month']}={_structure(row)}" for row in rows),
        "MIX_CHANGES|" + ("|".join(changes) or "none"),
    ))


# --- Pattern checks -----------------------------------------------------------
# The model names the kind of change it saw, from three broad labels per check (Qwen 7B
# invents or misuses names when given many). Whatever it writes is grouped into one of
# these by the words it uses, then confirmed against the rows. A mismatch is listed for
# staff under the model's sentence; nothing the model wrote is hidden.

PATTERN_NAMES: dict[str, tuple[str, ...]] = {
    "activity_and_amount_change": ("large_transaction",),
    "money_in_and_out": ("debits", "credits", "debit_credit_mix_change"),
    "burst_and_gaps": ("burst", "long_gap"),
}
# How each pattern is shown to staff.
PATTERN_LABELS: dict[str, str] = {
    "large_transaction": "Large transaction",
    "debits": "Debits (money out)",
    "credits": "Credits (money in)",
    "debit_credit_mix_change": "Debit/credit mix changed",
    "burst": "Burst",
    "long_gap": "Long gap",
}
# Words in a name the model wrote, and the pattern they mean; the first match wins.
_PATTERN_WORDS: dict[str, tuple[tuple[str, tuple[str, ...]], ...]] = {
    "activity_and_amount_change": (
        ("large_transaction", ("large", "single", "big")),
    ),
    "money_in_and_out": (
        ("debit_credit_mix_change", ("mix", "shift", "switch", "direction")),
        ("debits", ("debit", "money_out", "outflow")),
        ("credits", ("credit", "money_in", "inflow")),
    ),
    "burst_and_gaps": (
        ("burst", ("burst",)),
        ("long_gap", ("long", "before", "quiet", "inactiv", "dormant")),
    ),
}


def canonical_pattern(check: str, name: str) -> str:
    """One of the check's three patterns for what the model wrote; "none" stays "none".

    An unrecognised name is returned as written, so staff see it and its issue.
    """
    words = re.sub(r"[^a-z]+", "_", name.lower()).strip("_")
    if words in ("", "none"):
        return "none"
    if check == "money_in_and_out" and "debit" in words and "credit" in words:
        return "debit_credit_mix_change"  # e.g. "debit_only -> credit_only"
    for pattern, keys in _PATTERN_WORDS.get(check, ()):
        if any(key in words for key in keys):
            return pattern
    return name


def pattern_problem(
    check: str, pattern: str, rows: list[dict[str, Any]], months: list[str], *,
    long_gap_days: Decimal, single_reference: Decimal,
) -> str | None:
    """Why the pattern name does not fit the selected months, in plain words; None when it fits."""
    by_month = {row["year_month"]: row for row in rows}
    selected = [by_month[month] for month in months if month in by_month]
    if pattern == "none" or not selected:
        return "No pattern label was given for a found pattern."
    if pattern not in PATTERN_NAMES.get(check, ()):
        return f'"{pattern}" is not one of the pattern labels for this check.'
    fits = _pattern_fits(pattern, rows, selected, long_gap_days, single_reference)
    return None if fits else (
        f'"{PATTERN_LABELS[pattern]}" does not fit the selected months: {_selected_facts(check, rows, selected)}.'
    )


def _before_after(rows: list[dict[str, Any]], selected: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]] | None:
    """The two months being compared: the first and last selected, or one month and the month before it."""
    if len(selected) >= 2:
        return selected[0], selected[-1]
    index = rows.index(selected[0])
    return (rows[index - 1], selected[0]) if index > 0 else None


def _pattern_fits(
    pattern: str, rows: list[dict[str, Any]], selected: list[dict[str, Any]], long_gap_days: Decimal,
    single_reference: Decimal,
) -> bool:
    if pattern == "large_transaction":
        return any(_amount(row, "max_amount") >= single_reference for row in selected)
    if pattern == "burst":
        return bool(burst_guide_months(selected))
    if pattern == "long_gap":
        return bool(long_gap_months(selected, long_gap_days))
    pair = _before_after(rows, selected)
    if pair is None:
        return False
    before, after = pair
    compare = {
        # Debits/credits: that side changed, or both months had only that side.
        "debits": lambda: (_amount(after, "monthly_debit") != _amount(before, "monthly_debit")
                           or _structure(before) == _structure(after) == "debit_only"),
        "credits": lambda: (_amount(after, "monthly_credit") != _amount(before, "monthly_credit")
                            or _structure(before) == _structure(after) == "credit_only"),
        "debit_credit_mix_change": lambda: _structure(after) != _structure(before),
    }
    return compare[pattern]()


def _selected_facts(check: str, rows: list[dict[str, Any]], selected: list[dict[str, Any]]) -> str:
    """What the rows actually show for the selected months, for a mismatch message."""
    if check == "money_in_and_out":
        return "; ".join(f"{row['year_month']} has {_STRUCTURE_WORDS[_structure(row)]}" for row in selected)
    if check == "burst_and_gaps":
        return "; ".join(_burst_month_phrase(row) for row in selected)
    largest = max(rows, key=lambda row: _amount(row, "max_amount"))
    return "; ".join(
        f"{row['year_month']} total {_rm(_amount(row, 'total_amount'))}, largest single {_rm(_amount(row, 'max_amount'))}"
        for row in selected
    ) + f" (largest single in six months: {_rm(_amount(largest, 'max_amount'))} in {largest['year_month']})"


# --- Numbers a model sentence may quote ------------------------------------------

_SMALL_COUNTS = {Decimal(n) for n in range(0, 13)}
# A number as written in text: 1,234,567.89 (commas only between groups of three digits) or 202605.
# "202604,202605" is two months, not one number.
_NUMBER_IN_TEXT = re.compile(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?")


def _with_roundings(values: set[Decimal]) -> set[Decimal]:
    rounded: set[Decimal] = set()
    for value in values:
        rounded.add(value.normalize())
        for places in ("1", "0.1", "0.01"):
            rounded.add(value.quantize(Decimal(places)).normalize())
    return rounded


def quotable_numbers(rows: list[dict[str, Any]], months: list[str], fields: tuple[str, ...]) -> set[Decimal]:
    """Months, years, small counts, and the given fields of the given months."""
    selected = [row for row in rows if row["year_month"] in months] if months else rows
    values = set(_SMALL_COUNTS) | calendar_numbers(rows)
    for row in selected:
        values.update(Decimal(str(row[field])) for field in fields)
    return _with_roundings(values)


_MONTH_NUMBERS = {name: index for index, name in enumerate(
    ("jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"), start=1)}
# Capitalised month names only; "May" counts only with a year, so "staff may ask" is never read as a month.
_MONTH_NAME = re.compile(
    r"\b(?:(January|February|March|April|June|July|August|September|October|November|December"
    r"|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sept?|Oct|Nov|Dec)\.?(?:\s+(\d{4}))?|(May)\s+(\d{4}))\b"
)


def unmatched_month_name(text: str, months: list[str]) -> str | None:
    """The first month written as a name ("June 2025") that is not one of ``months``, or None.

    The number check cannot see a wrong month name next to a correct amount, e.g.
    "RM 10,000.00 in June 2025" for a 202511 transaction.
    """
    for match in _MONTH_NAME.finditer(text):
        name, year = (match.group(1), match.group(2)) if match.group(1) else (match.group(3), match.group(4))
        number = f"{_MONTH_NUMBERS[name[:3].lower()]:02d}"
        if not any(month[4:] == number and (year is None or month[:4] == year) for month in months):
            return match.group(0)
    return None


def calendar_numbers(rows: list[dict[str, Any]]) -> set[Decimal]:
    """The months (202604) and years (2026) of the rows, so "May 2026" is never read as a figure."""
    return {Decimal(value) for row in rows for value in (row["year_month"], row["year_month"][:4])}


def numbers_in_text(text: str) -> set[Decimal]:
    """Every number written in a prompt input, with roundings (for summary grounding)."""
    return _with_roundings({Decimal(token.replace(",", "")) for token in _NUMBER_IN_TEXT.findall(text)} | _SMALL_COUNTS)


def unquotable_number(text: str, allowed: set[Decimal]) -> str | None:
    """The first number in ``text`` that is not allowed, or None."""
    for token in _NUMBER_IN_TEXT.findall(text):
        if Decimal(token.replace(",", "")).normalize() not in allowed:
            return token
    return None


# --- Rows for the summary call ------------------------------------------------------

def summary_month_lines(rows: list[dict[str, Any]]) -> list[str]:
    """The summary's MONTH rows: call 1's labelled fields plus the largest single, burst share and gap."""
    lines = []
    for row in rows:
        extra = (
            f"|largest_single={_money(_amount(row, 'max_amount'))}"
            f"|burst_share={_burst_share(row)}%"
            f"|avg_days_between={_avg_days_cell(row)}"
        )
        lines.append(month_fact_line(row) + extra)
    return lines


def profile_activity_focus(rows: list[dict[str, Any]]) -> str:
    """The profile call's focus line: the biggest money in, money out and single transaction.

    No monthly total on purpose: Qwen reported the busiest month's total as credits
    (logged 2026-10-05, RM 7,234.04 called "credit transactions"). A total is not
    income; credits are what the declared occupation is compared with.
    """
    def peak(amount_field: str, count_field: str, noun: str) -> str:
        row = max(rows, key=lambda item: _amount(item, amount_field))
        if _amount(row, amount_field) == 0:
            return "none"
        return f"{row['year_month']}:{_rm(_amount(row, amount_field))} from {_count_label(row[count_field], noun)}"

    largest = max(rows, key=lambda item: _amount(item, "max_amount"))
    return (
        f"PROFILE_ACTIVITY_FOCUS|highest_credits={peak('monthly_credit', 'credit_count_monthly', 'credit')}|"
        f"highest_debits={peak('monthly_debit', 'debit_count_monthly', 'debit')}|"
        f"largest_single={largest['year_month']}:{_rm(_amount(largest, 'max_amount'))}"
    )