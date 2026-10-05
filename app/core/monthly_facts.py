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
    active = [row for row in rows if row["txn_count_monthly"] > 0]
    largest = max(rows, key=lambda row: _amount(row, "max_amount"))
    highest = max(rows, key=lambda row: _amount(row, "total_amount"))
    reference = "above" if _amount(largest, "max_amount") >= single_reference else "below"
    lines.append(
        f"AMOUNT_CANDIDATES|largest_single_in_six_months={largest['year_month']}:{_money(_amount(largest, 'max_amount'))}|"
        f"largest_single_vs_review_reference={reference} (reference {_money(single_reference)})|"
        f"highest_monthly_total={highest['year_month']}:{_money(_amount(highest, 'total_amount'))}|"
        f"active_months={len(active)}|"
        "total_rise=" + _directional_pair(rows, "total_amount", True)
        + "|total_fall=" + _directional_pair(rows, "total_amount", False)
        + "|std_rise=" + _directional_pair(rows, "std_amount", True)
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
    by_month = {row["year_month"]: row for row in rows}
    first, second = (by_month[month] for month in months)
    return (
        f"{months[0]}: {_count_label(first['debit_count_monthly'], 'debit')} {_rm(_amount(first, 'monthly_debit'))}, "
        f"{_count_label(first['credit_count_monthly'], 'credit')} {_rm(_amount(first, 'monthly_credit'))}; "
        f"{months[1]}: {_count_label(second['debit_count_monthly'], 'debit')} {_rm(_amount(second, 'monthly_debit'))}, "
        f"{_count_label(second['credit_count_monthly'], 'credit')} {_rm(_amount(second, 'monthly_credit'))}."
    )


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
# Guide given to the model for a notable burst month; the model still decides.
BURST_GUIDE_SHARE = Decimal("25.0")
BURST_GUIDE_TRANSACTIONS = 4


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


def has_burst(rows: list[dict[str, Any]]) -> bool:
    return any(_burst_share(row) > 0 for row in rows)


def has_in_month_gap(rows: list[dict[str, Any]]) -> bool:
    return any(gap_basis(row) == "in_month" for row in rows)


def burst_gap_facts(rows: list[dict[str, Any]]) -> str:
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
    burst = [row for row in rows if _burst_share(row) > 0]
    in_month = [row for row in rows if gap_basis(row) == "in_month"]
    since_previous = [row for row in rows if gap_basis(row) == "since_previous"]
    peak = max(burst, key=_burst_share) if burst else None
    longest = max(in_month, key=_gap_days) if in_month else None
    shortest = min(in_month, key=_gap_days) if in_month else None
    longest_back = max(since_previous, key=_gap_days) if since_previous else None
    rises = [(a, b) for a, b in zip(rows, rows[1:]) if _burst_share(b) > _burst_share(a)]
    rise = max(rises, key=lambda pair: _burst_share(pair[1]) - _burst_share(pair[0])) if rises else None
    guide = [row for row in burst if _burst_share(row) >= BURST_GUIDE_SHARE and row["txn_count_monthly"] >= BURST_GUIDE_TRANSACTIONS]
    lines.append(
        "BURST_GUIDE|months_with_burst_share_25%_or_more_and_4_or_more_transactions="
        + (",".join(f"{row['year_month']}:{_burst_share(row)}%" for row in guide) or "no month")
    )
    lines.append(
        "BURST_CANDIDATES|burst_months=" + (",".join(row["year_month"] for row in burst) or "none")
        + "|peak_burst=" + (f"{peak['year_month']}:{_burst_share(peak)}%" if peak else "none")
        + "|largest_burst_rise=" + (
            f"{rise[0]['year_month']},{rise[1]['year_month']}:{_burst_share(rise[0])}%->{_burst_share(rise[1])}%"
            if rise else "none")
        + "|longest_in_month_gap=" + (f"{longest['year_month']}:{_gap_days(longest)}" if longest else "none")
        + "|shortest_in_month_gap=" + (f"{shortest['year_month']}:{_gap_days(shortest)}" if shortest else "none")
        + "|longest_gap_to_previous=" + (
            f"{longest_back['year_month']}:{_gap_days(longest_back)}" if longest_back else "none")
    )
    return "\n".join(lines)


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
    burst = [row for row in rows if _burst_share(row) > 0]
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
    burst = [row for row in rows if _burst_share(row) > 0]
    in_month = [row for row in rows if gap_basis(row) == "in_month"]
    since_previous = [row for row in rows if gap_basis(row) == "since_previous"]
    if burst:
        peak = max(burst, key=_burst_share)
        burst_text = (
            f"Burst (same counterparty more than 3 times in a month) in {len(burst)} of {len(rows)} months, "
            f"peak {_burst_share(peak)}% in {peak['year_month']} ({_count_label(peak['txn_count_monthly'], 'transaction')})"
        )
    else:
        burst_text = "No burst (same counterparty more than 3 times in a month) in any month"
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


# --- Pattern checks -----------------------------------------------------------
# The model names the kind of change it saw. These checks confirm the name against
# the rows for the months it selected. A mismatch only hides the model's sentence
# (the outcome, months and table stay), so a wrong description never reaches staff.

def _structure(row: dict[str, Any]) -> str:
    debit, credit = row["debit_count_monthly"] > 0, row["credit_count_monthly"] > 0
    return "mixed" if debit and credit else "debit_only" if debit else "credit_only" if credit else "zero"


# Pattern names the model may give, per check, with the names Qwen has been seen to
# use instead. Unknown names are left as they are and fail the check below.
PATTERN_NAMES: dict[str, tuple[str, ...]] = {
    "activity_and_amount_change": (
        "large_single_amount", "total_rose", "total_fell", "amounts_more_varied", "started", "stopped",
    ),
    "money_in_and_out": (
        "debit_only", "credit_only", "debit_credit_mix_changed", "debit_amount_changed", "credit_amount_changed",
    ),
    "burst_and_gaps": ("burst_peak", "burst_rising", "gap_changed", "long_gap_before"),
}
_PATTERN_SYNONYMS: dict[str, dict[str, str]] = {
    "activity_and_amount_change": {
        "rose": "total_rose", "increase": "total_rose", "total_increase": "total_rose", "amount_rose": "total_rose",
        "fell": "total_fell", "decrease": "total_fell", "total_decrease": "total_fell", "amount_fell": "total_fell",
        "large_single": "large_single_amount", "large_transaction": "large_single_amount",
        "large_single_transaction": "large_single_amount", "std_rose": "amounts_more_varied",
        "more_varied": "amounts_more_varied",
    },
    "money_in_and_out": {
        "money_out_only": "debit_only", "debits_only": "debit_only", "one_sided_debit": "debit_only",
        "money_in_only": "credit_only", "credits_only": "credit_only", "one_sided_credit": "credit_only",
        "in_out_mix_changed": "debit_credit_mix_changed", "mix_changed": "debit_credit_mix_changed",
        "direction_shift": "debit_credit_mix_changed",
        "debit_increase": "debit_amount_changed", "debit_decrease": "debit_amount_changed",
        "debits_changed": "debit_amount_changed", "debit_outflow": "debit_amount_changed",
        "credit_increase": "credit_amount_changed", "credit_decrease": "credit_amount_changed",
        "credits_changed": "credit_amount_changed", "credit_inflow": "credit_amount_changed",
    },
    "burst_and_gaps": {"burst": "burst_peak", "long_gap": "long_gap_before", "gap_before": "long_gap_before"},
}


def canonical_pattern(check: str, name: str) -> str:
    """The standard pattern name for what the model wrote ("none" stays "none")."""
    return _PATTERN_SYNONYMS.get(check, {}).get(name, name)


def pattern_problem(check: str, pattern: str, rows: list[dict[str, Any]], months: list[str]) -> str | None:
    """Why the pattern name does not fit the selected months, or None when it fits.

    The model names the kind of change it saw; this confirms the name against the
    rows. A mismatch only hides the model's sentence (outcome, months and table stay).
    """
    by_month = {row["year_month"]: row for row in rows}
    selected = [by_month[month] for month in months if month in by_month]
    if pattern == "none" or not selected:
        return "no pattern named for a found pattern"
    if pattern not in PATTERN_NAMES.get(check, ()):
        return f"unknown pattern {pattern}"
    first, last = selected[0], selected[-1]
    pair = len(selected) == 2
    if check == "activity_and_amount_change":
        largest = max(_amount(row, "max_amount") for row in rows)
        fits = {
            "large_single_amount": any(_amount(row, "max_amount") == largest > 0 for row in selected),
            "total_rose": pair and _amount(last, "total_amount") > _amount(first, "total_amount"),
            "total_fell": pair and _amount(last, "total_amount") < _amount(first, "total_amount"),
            "amounts_more_varied": pair and _amount(last, "std_amount") > _amount(first, "std_amount"),
            "started": pair and first["txn_count_monthly"] == 0 and last["txn_count_monthly"] > 0,
            "stopped": pair and first["txn_count_monthly"] > 0 and last["txn_count_monthly"] == 0,
        }
    elif check == "money_in_and_out":
        fits = {
            "debit_only": all(_structure(row) == "debit_only" for row in selected),
            "credit_only": all(_structure(row) == "credit_only" for row in selected),
            "debit_credit_mix_changed": pair and _structure(first) != _structure(last),
            "debit_amount_changed": pair and _amount(first, "monthly_debit") != _amount(last, "monthly_debit"),
            "credit_amount_changed": pair and _amount(first, "monthly_credit") != _amount(last, "monthly_credit"),
        }
    else:
        peak = max((_burst_share(row) for row in rows), default=Decimal(0))
        fits = {
            "burst_peak": peak > 0 and any(_burst_share(row) == peak for row in selected),
            "burst_rising": pair and _burst_share(last) > _burst_share(first),
            "gap_changed": (pair and gap_basis(first) == gap_basis(last) == "in_month"
                            and _gap_days(first) != _gap_days(last)),
            "long_gap_before": any(gap_basis(row) == "since_previous" for row in selected),
        }
    return None if fits[pattern] else f"{pattern} does not match the selected months"


# --- Numbers a model sentence may quote ------------------------------------------

_SMALL_COUNTS = {Decimal(n) for n in range(0, 13)}
_NUMBER_IN_TEXT = re.compile(r"\d[\d,]*(?:\.\d+)?")


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
    lines = [
        "MONTH|month|transactions|total_RM|average_RM|std_RM|largest_single_RM|debit_count|debit_RM|"
        "credit_count|credit_RM|burst_%|avg_days_between_transactions"
    ]
    for row in rows:
        lines.append(
            f"MONTH|{row['year_month']}|{row['txn_count_monthly']}|{_money(_amount(row, 'total_amount'))}|"
            f"{_money(_amount(row, 'avg_amount'))}|{_money(_amount(row, 'std_amount'))}|"
            f"{_money(_amount(row, 'max_amount'))}|{row['debit_count_monthly']}|{_money(_amount(row, 'monthly_debit'))}|"
            f"{row['credit_count_monthly']}|{_money(_amount(row, 'monthly_credit'))}|{_burst_share(row)}%|{_avg_days_cell(row)}"
        )
    return lines