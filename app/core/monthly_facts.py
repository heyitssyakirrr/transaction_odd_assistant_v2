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


def _amount_phrase(row: dict[str, Any]) -> str:
    return (
        f"{_count_label(row['txn_count_monthly'], 'transaction')}, total {_rm(_amount(row, 'total_amount'))}, "
        f"largest single {_rm(_amount(row, 'max_amount'))}"
    )


def _share(part: Decimal, whole: Decimal) -> str:
    return f"{(part / whole * 100).quantize(Decimal('0.1'))}%" if whole else "0.0%"


# --- How this account usually transacts -------------------------------------
# No fixed amount decides what is large: the AI compares each month with this
# account's own usual level. The code does the arithmetic (7B models are weak at
# ratios) and gives the result as a figure; the AI makes the judgement.

# A usual level needs at least this many months with transactions; with fewer, a
# "middle value" would just be one or two transactions, so none is given.
_MIN_HISTORY_MONTHS = 3
_NOT_ENOUGH_HISTORY = "not_enough_history"


@dataclass(frozen=True)
class AccountUsual:
    """The middle value across the months with transactions; None with too little history."""

    active_months: int
    largest_single: Decimal | None
    month_total: Decimal | None


def account_usual(rows: list[dict[str, Any]]) -> AccountUsual:
    active = [row for row in rows if row["txn_count_monthly"] > 0]
    if len(active) < _MIN_HISTORY_MONTHS:
        return AccountUsual(len(active), None, None)
    return AccountUsual(
        len(active),
        _median([_amount(row, "max_amount") for row in active]),
        _median([_amount(row, "total_amount") for row in active]),
    )


def _median(values: list[Decimal]) -> Decimal:
    ordered = sorted(values)
    middle = len(ordered) // 2
    value = ordered[middle] if len(ordered) % 2 else (ordered[middle - 1] + ordered[middle]) / 2
    return value.quantize(Decimal("0.01"))


def _times(value: Decimal, usual: Decimal | None) -> str:
    """How many times the usual level, e.g. 7.7; not_enough_history when there is no usual level."""
    if not usual:
        return _NOT_ENOUGH_HISTORY
    ratio = value / usual
    # Two decimals under 1, so a small month reads 0.05 rather than 0.0 (which looks like nothing).
    return f"{ratio.quantize(Decimal('0.01') if ratio < 1 else Decimal('0.1'))}"


def account_usual_line(usual: AccountUsual) -> str:
    if usual.largest_single is None:
        return (f"ACCOUNT_USUAL|months_with_transactions={usual.active_months}|usual={_NOT_ENOUGH_HISTORY} "
                f"(fewer than {_MIN_HISTORY_MONTHS} months with transactions)")
    return (
        f"ACCOUNT_USUAL|months_with_transactions={usual.active_months}|"
        f"usual_largest_single={_money(usual.largest_single)}|usual_month_total={_money(usual.month_total)}"
    )


def _largest_single(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """The month with the largest single transaction of the six (the earliest on a tie)."""
    return max(rows, key=lambda row: _amount(row, "max_amount"))


def amount_facts(rows: list[dict[str, Any]], usual: AccountUsual) -> str:
    """Call 1 facts for the large-transaction check: each month's largest single transaction
    against this account's usual largest. The AI decides whether one is large."""
    lines = [
        "AMOUNT_FACTS — amounts in RM. times_usual_largest = this month's largest single transaction divided by "
        "usual_largest_single. share_of_month_total = the largest single transaction as a share of the month total."
    ]
    for row in rows:
        if not row["txn_count_monthly"]:
            lines.append(f"LARGEST|{row['year_month']}|transactions=0")
            continue
        largest = _amount(row, "max_amount")
        lines.append(
            f"LARGEST|{row['year_month']}|largest_single={_money(largest)}|"
            f"times_usual_largest={_times(largest, usual.largest_single)}|transactions={row['txn_count_monthly']}|"
            f"month_total={_money(_amount(row, 'total_amount'))}|"
            f"share_of_month_total={_share(largest, _amount(row, 'total_amount'))}"
        )
    top = _largest_single(rows)
    lines.append(
        f"LARGEST_IN_SIX_MONTHS|{top['year_month']}|largest_single={_money(_amount(top, 'max_amount'))}|"
        f"times_usual_largest={_times(_amount(top, 'max_amount'), usual.largest_single)}"
    )
    return "\n".join(lines)


def _vs_usual_largest(row: dict[str, Any], usual: AccountUsual) -> str:
    """", 7.7 times the usual largest (RM 17,511.50)" for staff; empty with too little history."""
    if usual.largest_single is None:
        return ""
    return (f", {_times(_amount(row, 'max_amount'), usual.largest_single)} times the usual largest "
            f"({_rm(usual.largest_single)})")


def activity_pair_context(rows: list[dict[str, Any]], months: list[str]) -> str:
    """Key figures for one or two selected months, led by the amounts."""
    by_month = {row["year_month"]: row for row in rows}
    if len(months) == 1:
        row = by_month[months[0]]
        return f"{months[0]}: {_amount_phrase(row)}{_vs_usual_largest(row, account_usual(rows))}."
    first, second = (by_month[month] for month in months)
    return (
        f"{months[0]} to {months[1]}: total {_rm(_amount(first, 'total_amount'))} to {_rm(_amount(second, 'total_amount'))}; "
        f"largest single {_rm(_amount(first, 'max_amount'))} to {_rm(_amount(second, 'max_amount'))}; "
        f"transactions {first['txn_count_monthly']} to {second['txn_count_monthly']}."
    )


def activity_six_month_context(rows: list[dict[str, Any]]) -> str:
    totals = [_amount(row, "total_amount") for row in rows]
    largest = _largest_single(rows)
    return (
        f"Six-month totals range {_rm(min(totals))} to {_rm(max(totals))}; largest single transaction "
        f"{_rm(_amount(largest, 'max_amount'))} in {largest['year_month']}{_vs_usual_largest(largest, account_usual(rows))}."
    )


# --- Activity after an inactive period -------------------------------------
# Whether this pattern exists is arithmetic on the six rows, so the code decides
# it. The model still judges the overall risk. Wording stays observational: six
# rows cannot establish formal dormancy or that the account was ever active.
# No reference amount: the amount is stated, and the AI judges what it means.

Direction = Literal["debit", "credit", "mixed", "unspecified"]


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
    later_active_months: int
    later_transactions: int
    later_total: Decimal


def find_inactivity_run(rows: list[dict[str, Any]], *, min_zero_months: int) -> InactivityRun | None:
    """Return the most recent run of ``min_zero_months`` or more zero-count months
    immediately followed by a month with transactions, or None.

    Rows are the six consecutive months in order (guaranteed by the CSV loader).
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
        later = [row for row in rows[index + 1:] if row["txn_count_monthly"] > 0]
        return InactivityRun(
            zero_start=rows[start]["year_month"], zero_end=rows[index - 1]["year_month"], zero_months=zeros,
            active_month=active["year_month"], transactions=active["txn_count_monthly"],
            debit_count=active["debit_count_monthly"], credit_count=active["credit_count_monthly"],
            total=total, debit=debit, credit=credit, largest_single=largest, direction=direction,
            later_active_months=len(later), later_transactions=sum(row["txn_count_monthly"] for row in later),
            later_total=sum((_amount(row, "total_amount") for row in later), Decimal(0)),
        )
    return None


def inactivity_run_fact(run: InactivityRun | None, min_zero_months: int) -> str:
    """One labelled prompt line. It states facts, never a risk level."""
    if run is None:
        return f"INACTIVITY_RUN|status=none|rule={min_zero_months}_or_more_zero_months_then_activity"
    return (
        f"INACTIVITY_RUN|status=present|zero_start={run.zero_start}|zero_end={run.zero_end}|"
        f"zero_months={run.zero_months}|active_month={run.active_month}|transactions={run.transactions}|"
        f"debits={run.debit_count}/{_money(run.debit)}|credits={run.credit_count}/{_money(run.credit)}|"
        f"largest_single={_money(run.largest_single)}|direction={run.direction}|"
        f"later_active_months={run.later_active_months}"
    )


def inactivity_summary_fact(run: InactivityRun | None, min_zero_months: int) -> str:
    """The inactive period for the summary: when, and the largest single transaction after it.

    No debit/credit direction: the summary does not describe flows.
    """
    if run is None:
        return f"INACTIVITY_RUN|status=none|rule={min_zero_months}_or_more_zero_months_then_activity"
    return (
        f"INACTIVITY_RUN|status=present|zero_months={run.zero_months}|zero_start={run.zero_start}|"
        f"zero_end={run.zero_end}|active_month={run.active_month}|transactions={run.transactions}|"
        f"largest_single_in_active_month={_money(run.largest_single)}"
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
    return {
        "debit": "Suggested check: confirm the payment purpose and how the account was funded.",
        "credit": "Suggested check: confirm the source and purpose of the incoming funds.",
    }.get(run.direction, "Suggested check: confirm the payment purpose and the source of incoming funds.")


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
# The prompt, the pattern check, the "says no burst" check and RISK_SIGNALS all use it.
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


# --- Layering: money in close to money out -----------------------------------
# Money passing through: in the same month, money in (credits) and money out (debits)
# are both large for this account and nearly equal. No fixed amount or percentage
# decides it: the code lays out the figures and the AI decides which months, if any,
# show layering, and explains why.

def _layering_difference(row: dict[str, Any]) -> Decimal:
    """How far apart money in and money out are, as a share of the larger (0 = equal)."""
    money_in, money_out = _amount(row, "monthly_credit"), _amount(row, "monthly_debit")
    larger = max(money_in, money_out)
    if not larger:
        return Decimal(100)
    return ((larger - min(money_in, money_out)) / larger * 100).quantize(Decimal("0.1"))


def _pct(value: Decimal) -> str:
    return f"{value.quantize(Decimal('0.1'))}%"


def _has_money_in_and_out(row: dict[str, Any]) -> bool:
    return _amount(row, "monthly_credit") > 0 and _amount(row, "monthly_debit") > 0


def _difference_cell(row: dict[str, Any]) -> str:
    """The difference for the evidence table; "-" for a month with no money in or out."""
    has_money = _amount(row, "monthly_credit") or _amount(row, "monthly_debit")
    return _pct(_layering_difference(row)) if has_money else "-"


def _difference_fact(row: dict[str, Any]) -> str:
    """The difference for the AI, in RM and as a share of the larger side; one-sided months are named."""
    money_in, money_out = _amount(row, "monthly_credit"), _amount(row, "monthly_debit")
    if not money_in and not money_out:
        return "no_money_in_or_out"
    if not money_in or not money_out:
        return "money_out_only" if not money_in else "money_in_only"
    return f"{_money(abs(money_in - money_out))} ({_pct(_layering_difference(row))})"


def closest_money_in_and_out(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    """The month whose money in and money out are nearest each other (both present); None if no month has both.
    A figure for the AI and staff to consider, not a finding."""
    both = [row for row in rows if _has_money_in_and_out(row)]
    return min(both, key=_layering_difference) if both else None


def _layering_phrase(row: dict[str, Any]) -> str:
    return (
        f"{row['year_month']}: money in {_rm(_amount(row, 'monthly_credit'))}, "
        f"money out {_rm(_amount(row, 'monthly_debit'))}, difference {_pct(_layering_difference(row))}"
    )


def layering_facts(rows: list[dict[str, Any]], usual: AccountUsual) -> str:
    """Call 1 facts for the layering check: money in and out per month, how close they are,
    and how big the month is for this account. The AI decides which months, if any, are layering."""
    lines = [
        "LAYERING_FACTS — money in = credits, money out = debits. difference = the gap between money in and money "
        "out, in RM and as a share of the larger side (0% = equal). times_usual_month = this month's total divided "
        "by usual_month_total."
    ]
    for row in rows:
        lines.append(
            f"LAYERING|{row['year_month']}|money_in={_money(_amount(row, 'monthly_credit'))}|"
            f"money_out={_money(_amount(row, 'monthly_debit'))}|difference={_difference_fact(row)}|"
            f"month_total={_money(_amount(row, 'total_amount'))}|"
            f"times_usual_month={_times(_amount(row, 'total_amount'), usual.month_total)}"
        )
    closest = closest_money_in_and_out(rows)
    lines.append("CLOSEST_MONEY_IN_AND_OUT|" + (
        f"{closest['year_month']}|difference={_difference_fact(closest)}" if closest
        else "no month has both money in and money out"))
    return "\n".join(lines)


def layering_context(rows: list[dict[str, Any]], months: list[str]) -> str:
    """Key figures for the layering check, from the rows: the months the AI judged layering,
    or, when it judged none, the month where money in and money out were closest.

    Full figures for one or two months; month and difference only for more, so the text
    stays short.
    """
    by_month = {row["year_month"]: row for row in rows}
    selected = [by_month[month] for month in months if month in by_month]
    if len(selected) > 2:
        return f"Months judged layering: {_short_layering_list(selected)}."
    if selected:
        return "; ".join(_layering_phrase(row) for row in selected) + "."
    closest = closest_money_in_and_out(rows)
    if closest is None:
        return "No month had both money in and money out."
    return f"No month judged layering. Closest money in and money out: {_layering_phrase(closest)}."


def _short_layering_list(rows: list[dict[str, Any]]) -> str:
    return ", ".join(f"{row['year_month']} (difference {_pct(_layering_difference(row))})" for row in rows)


# --- What the summary's risk guide reads -------------------------------------------

@dataclass(frozen=True)
class RiskSignals:
    """What call 1's AI decided: its large single transactions and its layering months.
    None means that check could not be verified. Facts only; the AI decides the risk level."""

    large_singles: tuple[str, ...] | None
    layering: tuple[str, ...] | None


def risk_signals(
    rows: list[dict[str, Any]], large_months: list[str] | None, layering_months: list[str] | None,
) -> RiskSignals:
    """The AI's decided months, each with its figures from the rows (None = not checked)."""
    by_month = {row["year_month"]: row for row in rows}
    return RiskSignals(
        large_singles=None if large_months is None else tuple(
            f"{month}: {_rm(_amount(by_month[month], 'max_amount'))}" for month in large_months),
        layering=None if layering_months is None else tuple(
            _layering_phrase(by_month[month]) for month in layering_months),
    )


def risk_signals_line(signals: RiskSignals) -> str:
    """The summary's data line for the risk guide; it lists the checks' decisions, never a level."""
    def listed(values: tuple[str, ...] | None) -> str:
        return "not_checked" if values is None else "; ".join(values) or "none"
    return (f"RISK_SIGNALS|large_single_transactions={listed(signals.large_singles)}|"
            f"layering_months={listed(signals.layering)}")


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
        ("Money in, credits (RM)", lambda r: _money(_amount(r, "monthly_credit"))),
        ("Money out, debits (RM)", lambda r: _money(_amount(r, "monthly_debit"))),
        ("Difference between money in and out", _difference_cell),
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


# --- Pattern checks -----------------------------------------------------------
# The model names the pattern it saw, from one or two labels per check (Qwen 7B invents
# or misuses names when given many). Whatever it writes is grouped into one of these by
# the words it uses, then confirmed against the rows. A mismatch is listed for
# staff under the model's sentence; nothing the model wrote is hidden.

PATTERN_NAMES: dict[str, tuple[str, ...]] = {
    "activity_and_amount_change": ("large_transaction",),
    "money_in_and_out": ("layering",),
    "burst_and_gaps": ("burst", "long_gap"),
}
# How each pattern is shown to staff.
PATTERN_LABELS: dict[str, str] = {
    "large_transaction": "Large transaction",
    "layering": "Layering (money in close to money out)",
    "burst": "Burst",
    "long_gap": "Long gap",
}
# Words in a name the model wrote, and the pattern they mean; the first match wins.
_PATTERN_WORDS: dict[str, tuple[tuple[str, tuple[str, ...]], ...]] = {
    "activity_and_amount_change": (
        ("large_transaction", ("large", "single", "big")),
    ),
    "money_in_and_out": (
        ("layering", ("layer", "pass", "through", "equal", "similar", "match", "close")),
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
    for pattern, keys in _PATTERN_WORDS.get(check, ()):
        if any(key in words for key in keys):
            return pattern
    return name


def pattern_problem(
    check: str, pattern: str, rows: list[dict[str, Any]], months: list[str], *, long_gap_days: Decimal,
) -> str | None:
    """Why the pattern name does not fit the selected months, in plain words; None when it fits.

    Large transaction and layering are the AI's judgement, so only what the rows make
    impossible is checked: a large transaction needs a transaction in that month, and
    layering needs both money in and money out in it.
    """
    by_month = {row["year_month"]: row for row in rows}
    selected = [by_month[month] for month in months if month in by_month]
    if pattern == "none" or not selected:
        return "No pattern label was given for a found pattern."
    if pattern not in PATTERN_NAMES.get(check, ()):
        return f'"{pattern}" is not one of the pattern labels for this check.'
    if pattern == "layering":
        one_sided = [row for row in selected if not _has_money_in_and_out(row)]
        return None if not one_sided else (
            f'"{PATTERN_LABELS[pattern]}" does not fit {"; ".join(_layering_phrase(row) for row in one_sided)} '
            "(layering needs both money in and money out in the month)."
        )
    fits = {
        "large_transaction": lambda: all(row["txn_count_monthly"] > 0 for row in selected),
        "burst": lambda: bool(burst_guide_months(selected)),
        "long_gap": lambda: bool(long_gap_months(selected, long_gap_days)),
    }[pattern]()
    return None if fits else (
        f'"{PATTERN_LABELS[pattern]}" does not fit the selected months: {_selected_facts(check, rows, selected)}.'
    )


def _selected_facts(check: str, rows: list[dict[str, Any]], selected: list[dict[str, Any]]) -> str:
    """What the rows actually show for the selected months, for a mismatch message."""
    if check == "burst_and_gaps":
        return "; ".join(_burst_month_phrase(row) for row in selected)
    largest = _largest_single(rows)
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
    """The summary's MONTH rows as labelled fields (Qwen 7B miscounts bare ``|`` columns).

    No debit or credit fields: money in and money out reach the summary only through
    the layering months, so the summary does not describe debit or credit flows.
    """
    return [
        f"MONTH|{row['year_month']}|transactions={row['txn_count_monthly']}|"
        f"total={_money(_amount(row, 'total_amount'))}|largest_single={_money(_amount(row, 'max_amount'))}|"
        f"burst_share={_burst_share(row)}%|avg_days_between={_avg_days_cell(row)}"
        for row in rows
    ]


def profile_activity_focus(rows: list[dict[str, Any]]) -> str:
    """The profile call's focus line: the largest single transaction against this account's usual largest.

    Only single transactions, on purpose. Given a month total, Qwen called it credits
    (logged 2026-10-05, RM 7,234.04); given monthly credits, it compared them with a
    single-transaction amount (logged 2026-10-08, RM 6,404.50 from 3 credits).
    """
    largest = _largest_single(rows)
    amount = _amount(largest, "max_amount")
    usual = account_usual(rows).largest_single
    return (
        f"PROFILE_ACTIVITY_FOCUS|largest_single_transaction={_rm(amount)} in {largest['year_month']}|"
        + (f"usual_largest_single={_rm(usual)}|times_usual_largest={_times(amount, usual)}" if usual
           else f"usual_largest_single={_NOT_ENOUGH_HISTORY}")
    )
