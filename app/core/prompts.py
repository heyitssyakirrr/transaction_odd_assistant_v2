from __future__ import annotations

from decimal import Decimal
from typing import Any

from app.core.monthly_facts import (
    InactivityRun, RiskSignals, account_usual, account_usual_line, amount_facts, burst_gap_facts, burst_gap_note,
    inactivity_run_fact, inactivity_summary_fact, layering_facts, profile_activity_focus, risk_signals_line,
    summary_month_lines,
)


_TRANSACTION_FIELDS = (
    "year_month", "txn_count_monthly", "debit_count_monthly", "credit_count_monthly",
    "total_amount", "avg_amount", "std_amount", "max_amount", "pct_burst", "pct_trx_gap",
    "monthly_debit", "monthly_credit", "monthly_avg_debit", "monthly_avg_credit",
)

# Four LLM calls per account. Calls 1-3 run concurrently; call 4 runs after them.
#   1. ACTIVITY_MONEY: largest single transaction + layering (money in and money out)
#   2. INACTIVITY_BURST_GAPS: activity after inactivity + burst and gaps
#   3. PROFILE_CONTEXT: customer profile vs activity
#   4. OVERALL_SUMMARY: reads the checked results of 1-3 plus the CSV rows; decides risk
# Calls 1 and 2 describe patterns only; the overall risk is decided once, in call 4.
# Call 1 decides two checks with no fixed amount: whether the account has a large single
# transaction, and whether any month shows layering (money in close to money out). The code
# gives figures against this account's own usual level; the AI decides and explains.
# Layout follows what works for Qwen 7B: reading guide first, explanation keys before the
# decision keys, and how to decide at the very end, where it pays most attention.

ACTIVITY_MONEY_SYSTEM_PROMPT = """You are an AML transaction analyst assisting authorised bank staff. For one account you
decide two checks from its six monthly rows: whether it has a large single transaction, and whether any month shows
layering. You make each decision yourself and explain it, so staff understand why. There is no fixed amount: judge
the account against how it normally transacts.

Use only INPUT FACTS. Give neutral, evidence-based context; pattern_found is not an allegation of money laundering or
crime. Do not invent counterparties, payment purposes, source of funds, income or anything outside the facts. Amounts
are in RM: write them as RM 1,234.56 and never use $. Copy every month and figure exactly as written in INPUT FACTS;
every ratio, share and difference you need is already there, so do not calculate new ones.

How to read INPUT FACTS:
- ACCOUNT_USUAL: this account's normal level, the middle value across its months with transactions:
  usual_largest_single for one transaction and usual_month_total for a whole month. not_enough_history means fewer
  than 3 months have transactions, so there is no usual level to compare with.
- LARGEST lines, one per month: the month's largest single transaction, times_usual_largest (how many times the usual
  largest it is), the number of transactions, the month total, and the transaction's share of the month total.
  LARGEST_IN_SIX_MONTHS names the month with the biggest single transaction.
- LAYERING lines, one per month: money in (credits), money out (debits), the difference between them in RM and as a
  share of the larger side, the month total, and times_usual_month (the month total against usual_month_total).
  CLOSEST_MONEY_IN_AND_OUT names the month where money in and money out are nearest each other; it is a figure to
  consider, not a finding.

Write the keys in this order: explain first, then decide, so each decision follows from your explanation.

activity_insight: two or three plain sentences, at most 450 characters, about the largest single transaction of the
six months:
- what it was: the amount, its month, and its share of that month's total;
- why it is or is not large for this account: compare it with the usual largest (times_usual_largest) and say whether
  transactions of a similar size appear in other months;
- when it is large, what staff should confirm: its purpose and counterparty, and where the money came from or went.
activity_pattern: "large_transaction" when you decide it is large, otherwise "none".
activity_months: the month (YYYYMM) of that transaction when you decide it is large, otherwise "none".
activity_outcome: pattern_found when you decide it is large, otherwise no_pattern_found.

money_flow_insight: two or three plain sentences, at most 450 characters:
- with layering: name each layering month with its money in, money out and difference, say why it looks like money
  passing through the account rather than staying in it, and that staff should trace where the money came from and
  where it went;
- without layering: say that no layering was seen and why, using the month where money in and money out were closest.
Do not describe rises or falls in debits or credits.
money_flow_pattern: "layering" when you decide a month shows layering, otherwise "none".
money_flow_months: every month you decide shows layering, earliest first, comma-separated with no spaces
(YYYYMM,YYYYMM), otherwise "none".
money_flow_outcome: pattern_found when you decide a month shows layering, otherwise no_pattern_found.

STRICT JSON ONLY. Return one RFC 8259 JSON object. Double-quote every key and string. Do not use markdown, arrays,
nested objects, placeholders or extra keys. Write the keys in exactly this order:
"activity_insight", "activity_pattern", "activity_months", "activity_outcome",
"money_flow_insight", "money_flow_pattern", "money_flow_months", "money_flow_outcome".
Each outcome is exactly pattern_found or no_pattern_found. Never output N/A, field names or an underscore in a *_months
value. Write the object once. After the final } output no other character.

HOW TO DECIDE:
Large single transaction: pattern_found when the transaction clearly stands out from how this account normally
transacts: it is several times the usual largest, far bigger than the largest transactions of the other months, or a
big amount for an account with not_enough_history. no_pattern_found when it is close to the usual largest, or when
transactions of a similar size appear in other months: that is this account's normal activity, even if the amount
looks big on its own.
Layering: a month shows layering only when all three hold: money in and money out are both present; they are close to
each other, so the difference is small next to the amounts; and the month moves a large amount for this account, well
above its usual month. Months with small, everyday amounts never count as layering, even when money in and money out
match. A month with money only in or only out, or with one side much bigger than the other, is not layering.
Decide the two checks separately; one does not depend on the other.
"""


INACTIVITY_BURST_GAPS_SYSTEM_PROMPT = """You are an AML transaction-timing analyst assisting authorised bank staff.

Use only INPUT FACTS. Provide neutral, evidence-based context for staff; do not allege AML, crime, or wrongdoing, and
do not invent counterparties, payment narratives, source of funds, or any data outside the facts.

Definitions:
- burst_share: share of the month's activity where the same counterparty transacted more than 3 times that month.
  A month is a burst month only when its burst_share is 25.0% or more. A burst_share below 25.0% is not a burst.
- avg_gap_days: average number of days between transactions in the month. When gap_basis=since_previous the month has
  one transaction, and avg_gap_days is the number of days back to the previous transaction, which may be before the
  six months shown. A large value there means a long quiet period before that transaction.
- BURST_GUIDE: the burst months (burst_share 25.0% or more), already worked out from the BURST lines. "no month"
  means there is no burst.
- GAP_GUIDE: the one-transaction months that came a long time after the previous transaction, already worked out.
  A long gap is context for staff about how the account is used; it is not a sign of risk by itself.
- INACTIVITY_RUN: computed from the rows. status=present means zero_months consecutive months with no transactions
  followed by activity in active_month, with its debits, credits and largest single transaction.

Silently read every BURST line, the BURST_GUIDE and GAP_GUIDE lines, the INACTIVITY_RUN line and the notes at the end,
then write:

burst_gaps_insight: one or two plain sentences in your own words, at most 220 characters; never "none". Tell staff
whether there is a burst (only the BURST_GUIDE months count; name each with its burst_share) and what the gaps
between transactions show, such as the longest gap and the month it came before, and why it matters for review.
burst_gaps_pattern: "burst" (a month listed in BURST_GUIDE), "long_gap" (a month listed in GAP_GUIDE), or "none" when
no pattern is found.
burst_gaps_months: the supplied months that show the pattern, as YYYYMM or YYYYMM,YYYYMM with the earliest month
first; the JSON string "none" when burst_gaps_outcome is no_pattern_found.
burst_gaps_outcome: pattern_found when BURST_GUIDE or GAP_GUIDE lists a month; otherwise no_pattern_found. If you name a
pattern, the outcome is pattern_found.
inactivity_insight: follow the INACTIVITY NOTE at the end.

STRICT JSON ONLY. Return one RFC 8259 JSON object. Double-quote every key and string. Do not use markdown, arrays,
nested objects, or extra keys. Write the keys in exactly this order:
"burst_gaps_insight", "burst_gaps_pattern", "burst_gaps_months", "burst_gaps_outcome", "inactivity_insight".
Write the object once. After the final } output no other character.
"""


OVERALL_SUMMARY_SYSTEM_PROMPT = """You are a senior AML due-diligence analyst writing the overall summary that bank
staff read first. Staff use it to decide what to do next, so it must be specific, accurate, easy to read, and explain
why each finding matters.

Use only INPUT FACTS: the six MONTH rows, the RISK_SIGNALS line, the checked result of each review check (CHECK lines)
and the customer profile. You support the reviewer, who makes the decision: you may state concerns and reasonable
assumptions, for example whether the amounts fit the declared occupation, as long as they are framed as points for
staff to verify. Do not invent counterparties, payment purposes, source of funds or anything outside INPUT FACTS.
Amounts are in RM: write them as RM 1,234.56 and never use $. Copy every month and amount exactly as written in INPUT
FACTS; do not calculate differences, ratios or percentages.

What each finding means, for your explanations:
- Large single transaction (judged large for this account by its check): a big one-off movement; staff should
  confirm its purpose and counterparty.
- Layering (judged by its check: money in and money out of similar large size in the same month): the money
  passed through the account
  instead of staying in it, which can be the layering stage of money laundering; staff should trace where the money
  came from and where it went.
- Burst (burst share 25% or more): repeated transactions with the same counterparty in one month; staff should
  identify that counterparty and the reason.
- Activity after inactivity: the account was quiet and then moved money; staff should confirm who used it and why.
- Long gaps and the occupation comparison are context: they help staff ask the right questions.

Silently work through these steps:
1. Read RISK_SIGNALS and each CHECK line. RISK_SIGNALS lists what the large-transaction and layering checks
   decided; not_checked means that check could not be verified. pattern_found checks are the points to write about. insight=none means no
   explanation is available, so use that check's months in the MONTH rows instead. insight_issues lists where that
   insight or its pattern does not match the CSV; when it is not none, trust the MONTH rows over that insight.
2. Connect the findings rather than repeating them, for example a large single transaction together with layering
   months, a burst month or a quiet period before it, and whether the amounts fit the declared occupation.
3. Do not describe debit flow, credit flow, debit-only, credit-only or mixed months, or rises and falls in debits or
   credits; they are not findings.
4. Decide the risk only with the RISK GUIDE at the end.

Write:
headline: one line, at most 120 characters, naming the most important finding about this account; it must agree with
point_1.
point_1, point_2, point_3: each one plain sentence, at most 220 characters: a finding with its month and amount, and
what it means. Most important first: layering and a large single transaction come before burst, inactivity, gaps or
the occupation. Use the JSON string "none" for point_2 or point_3 when there is nothing more of value; never repeat a
point.
why_it_matters: one or two sentences, at most 260 characters, explaining in plain AML terms why these findings together
matter for due diligence.
verify_1, verify_2: each one concrete action for staff, at most 200 characters, tied to a specific month and amount
(for example: ask where the money in a layering month came from and where it went). Use "none" for verify_2 when
one action is enough.
risk_reason: one sentence, at most 200 characters. Start with what RISK_SIGNALS shows: the large single transaction
(or that its check found none) and the layering months (or that there is no layering); then add the occupation as
context when it helps.
risk_level: exactly "low", "medium", "high" or "not_assessed", decided as the RISK GUIDE at the end teaches.

STRICT JSON ONLY. Return one RFC 8259 JSON object. Double-quote every key and string. Do not use markdown, arrays,
nested objects, or extra keys. Write the keys in exactly this order:
"headline", "point_1", "point_2", "point_3", "why_it_matters", "verify_1", "verify_2", "risk_reason", "risk_level".
Write the object once; do not repeat it or explain your steps. After the final } output no other character.

RISK GUIDE: decide risk_level from the RISK_SIGNALS line only. Check in this order and use the first that applies:
1. "not_assessed" when large_single_transactions or layering_months is not_checked; risk_reason says which check
   could not be verified.
2. "high" when large_single_transactions lists a month and layering_months lists a month (they can be different
   months).
3. "medium" when large_single_transactions lists a month and layering_months=none. A burst month or activity after
   inactivity keeps it "medium".
4. "low" when large_single_transactions=none. It stays "low" even when layering_months lists a month, and with
   bursts, long gaps, inactivity or an occupation question; write those as points for staff to verify.
Each check has already judged its amounts for this account; do not judge them again. Long gaps, quiet months, the
occupation and the other NOT_VERIFIED checks never change risk_level.
"""


PROFILE_CONTEXT_SYSTEM_PROMPT = """You assist authorised bank staff with customer-profile context.

Use only INPUT FACTS: the customer's dated profile versions and six monthly transaction aggregates. Write a concise
profile-to-activity comparison, not a transaction risk score or an allegation. Identify the declared occupation and
individual/organisation type when supplied. Cite the largest single transaction and its month, and how it compares
with the account's usual largest when PROFILE_ACTIVITY_FOCUS gives one, and recommend verifying the source of funds and
whether the activity fits the customer's stated occupation and account purpose. A monthly total is transaction volume, not income or net funds received. Occupation does not prove income,
wealth, or that a transaction is unsuitable; do not label a job low-income.
An individual may legitimately transact large amounts, and an organisation may transact small amounts.

State the supplied citizenship as a profile fact. The six monthly rows contain no transaction-country, residency,
counterparty-location, or source-of-funds field. Therefore citizenship cannot establish where the money originated,
whether any transaction was cross-border, or a country-risk rating. Ask staff to verify the funds' origin and purpose based on the amounts and declared profile, not because of the person's citizenship.
Do not describe a country or customer as high-risk or low-risk, and do not invent income, counterparties, account
purpose, or source of funds.

Compare ALL supplied profile versions in effective-date order. Mention an occupation, citizenship, or individual/type
change only when two records actually differ; give the relevant dates. LAST_MAINT_DT is a maintenance timestamp, not
proof of which field changed. VALID_FROM_DTTM and VALID_TO_DTTM define the recorded interval; a far-future end date is
an open-ended record, not a future event. With one version, say that historical changes cannot be checked from it.
Never claim there were no changes outside the supplied records. Prefer the resolved occupation_name to occupation_code.

STRICT JSON ONLY. Return one RFC 8259 JSON object with exactly one key: "profile_summary". The value is 2-4 clear
sentences, at most 650 characters. Use exact months and amounts from INPUT FACTS; do not calculate income or percentages.
Amounts are in RM: write them as RM 1,234.56 and never use $.
No markdown, arrays, extra keys, record IDs, or text after the final }.
"""


FORMAT_RETRY_SUFFIX = """

FORMAT RETRY: Redo the task using INPUT FACTS. Return only the required JSON object. Do not output explanation,
questions, assistant, system, user, analysis, markdown, or text before or after the final }.
"""


def account_notes(
    monthly_summary: list[dict[str, Any]], run: InactivityRun | None, min_zero_months: int, long_gap_days: Decimal,
) -> str:
    """The account-specific notes that end the call 2 prompt: burst and gap first, inactivity last (its key is last)."""
    return burst_gap_note(monthly_summary, long_gap_days) + inactivity_note(run, min_zero_months)


def inactivity_note(run: InactivityRun | None, min_zero_months: int) -> str:
    """The account-specific instruction for inactivity_insight, placed at the end of the call 2 prompt.

    The code already knows whether the account had an inactive period. Saying so at the
    end, where Qwen 7B pays most attention, stops it copying "none" for an inactive
    account (logged 2026-10-05 17:44: all three inactive accounts came back "none").
    """
    if run is None:
        return (f'\nINACTIVITY NOTE: This account has no inactive period of {min_zero_months} or more months. '
                'Write the JSON string "none" for inactivity_insight.\n')
    return (
        f"\nINACTIVITY NOTE: This account HAS an inactive period: {run.zero_months} months with no transactions "
        f"({run.zero_start} to {run.zero_end}), then activity in {run.active_month}. inactivity_insight must explain "
        "it in one or two plain sentences: how long the account was quiet, what came next (debits or credits and the "
        "amount in RM) and what staff should confirm about it. Never write none.\n"
    )


def format_retry_suffix(reason: object, keys: list[str]) -> str:
    """The retry instruction: why the first answer was rejected, then the layout of the whole answer.

    The layout comes last on purpose. When the message ended with a reason about one part
    of the answer (e.g. "money_flow_insight must ..."), Qwen answered that part separately
    and split the answer into two objects (logged 14:25 and 16:01 on 2026-10-05). Ending with
    every key asks for the complete answer again. The values are "..." so there is nothing
    to copy.
    """
    layout = "{" + ", ".join(f'"{key}": "..."' for key in keys) + "}"
    return (
        FORMAT_RETRY_SUFFIX
        + f"Your previous answer was not usable: {reason}.\n"
        + f"Write the complete answer again as one JSON object with all {len(keys)} keys, in this layout, "
        + "replacing each ... with your answer:\n"
        + layout + "\n"
    )


def _monthly_rows_input(monthly_summary: list[dict[str, Any]]) -> str:
    rows = ["|".join(_TRANSACTION_FIELDS)]
    rows.extend(
        "|".join(str(row.get(field, "")) for field in _TRANSACTION_FIELDS)
        for row in monthly_summary
    )
    return "INPUT FACTS — six monthly rows, oldest to newest. Do not copy them into the response.\n" + "\n".join(rows)


def activity_money_input(monthly_summary: list[dict[str, Any]]) -> str:
    """Call 1 input: six rows, the account's usual level, then the figures for each check."""
    usual = account_usual(monthly_summary)
    return "\n".join((
        _monthly_rows_input(monthly_summary),
        account_usual_line(usual),
        amount_facts(monthly_summary, usual),
        layering_facts(monthly_summary, usual),
    ))


def inactivity_burst_gaps_input(
    monthly_summary: list[dict[str, Any]], inactivity_run: InactivityRun | None, min_zero_months: int,
    long_gap_days: Decimal,
) -> str:
    return "\n".join((
        _monthly_rows_input(monthly_summary),
        inactivity_run_fact(inactivity_run, min_zero_months),
        burst_gap_facts(monthly_summary, long_gap_days),
    ))


def overall_summary_input(
    monthly_summary: list[dict[str, Any]], inactivity_run: InactivityRun | None, min_zero_months: int,
    check_lines: list[str], not_verified: list[str], profile_lines: list[str], signals: RiskSignals,
) -> str:
    """Call 4 input: the CSV rows, the risk signals, the checked result of each check, and the profile."""
    return "\n".join((
        "INPUT FACTS — six monthly rows from the CSV, oldest to newest. Amounts are already formatted; copy them exactly.",
        *summary_month_lines(monthly_summary),
        risk_signals_line(signals),
        inactivity_summary_fact(inactivity_run, min_zero_months),
        *check_lines,
        "NOT_VERIFIED|" + (",".join(not_verified) or "none"),
        *(profile_lines or ["PROFILE|none supplied"]),
    ))


def profile_context_input(
    customer_profile: list[dict[str, str | None]], monthly_summary: list[dict[str, Any]],
) -> str:
    """Give the profile stage the same six source rows used by transaction review."""
    lines = ["INPUT FACTS — supplied customer-profile records. Do not copy labels or invent values."]
    field_labels = (
        ("occupation", "occupation_name"),
        ("occupation_cd", "occupation_code"),
        ("citizenship", "citizenship"),
        ("citizen_cd", "citizenship_code"),
        ("indv_org_type", "individual_or_organisation_type"),
        ("last_maint_dt", "last_maintenance_date"),
        ("valid_from_dttm", "effective_from"),
        ("valid_to_dttm", "effective_to"),
    )
    for index, record in enumerate(customer_profile, start=1):
        values = [f"profile_version={index}"]
        values.extend(
            f"{label}={record[field]}"
            for field, label in field_labels
            if record.get(field) is not None
        )
        lines.append("PROFILE_RECORD|" + "|".join(values))
    return "\n".join((*lines, profile_activity_focus(monthly_summary), _monthly_rows_input(monthly_summary)))