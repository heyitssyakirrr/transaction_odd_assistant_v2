from __future__ import annotations

from decimal import Decimal
from typing import Any

from app.core.monthly_facts import (
    InactivityRun, RiskSignals, Screening, account_usual, account_usual_line, amount_facts, burst_gap_facts,
    burst_gap_note, inactivity_run_fact, inactivity_summary_fact, layering_facts, layering_note, profile_activity_focus,
    risk_signals_line, summary_month_lines, unusual_transaction_note,
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
# Calls 1-3 gather and describe evidence; only call 4 judges. In call 1 the code screens the
# account against its own usual level (app/core/monthly_facts.py) and states the finding in a
# note at the end of the prompt, where Qwen 7B pays most attention; the AI describes it. A 7B
# model asked to judge the figures itself called RM 8.00 "large" and every month "layering"
# (logged 2026-10-12), so call 1 has no decision to make and only two keys to write.

ACTIVITY_MONEY_SYSTEM_PROMPT = """You describe transaction evidence for authorised bank staff. The screening has
already checked this account; its findings are in the notes at the end. Describe that evidence clearly, using the
figures in INPUT FACTS. Do not say whether anything is suspicious and do not judge the customer: a later step does.

Use only INPUT FACTS and the notes. Do not invent counterparties, purposes, source of funds or income. Amounts are in
RM: write RM 1,234.56, never $. Copy months and figures exactly as written; do not calculate.

INPUT FACTS: ACCOUNT_USUAL is the account's normal level, the middle value of its months with transactions. LARGEST
lines give each month's largest single transaction and times_usual_largest. LAYERING lines give each month's money in
(credits), money out (debits), their difference and times_usual_month.

activity_insight: two plain sentences, at most 300 characters, describing what the UNUSUAL TRANSACTION NOTE found.
money_flow_insight: two plain sentences, at most 300 characters, describing what the LAYERING NOTE found. Never
describe rises or falls in debits or credits.

STRICT JSON ONLY. Return one RFC 8259 JSON object with exactly two keys, in this order: "activity_insight",
"money_flow_insight". Double-quote every key and string. No markdown, arrays or extra keys. Write the object once.
After the final } output no other character.
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

Use only INPUT FACTS: the six MONTH rows, the RISK_SIGNALS line, the evidence of each review check (CHECK lines)
and the customer profile. The checks only describe evidence; you are the one who judges it. You support the reviewer, who makes the decision: you may state concerns and reasonable
assumptions, for example whether the amounts fit the declared occupation, as long as they are framed as points for
staff to verify. Do not invent counterparties, payment purposes, source of funds or anything outside INPUT FACTS.
Amounts are in RM: write them as RM 1,234.56 and never use $. Copy every month and amount exactly as written in INPUT
FACTS; do not calculate differences, ratios or percentages.

What each finding means, for your explanations:
- Unusual transaction (listed in unusual_transaction): far above this account's usual size; staff should confirm its
  purpose and counterparty. The largest transaction of the six months is not a finding unless unusual_transaction
  lists it.
- Layering (listed in layering_months: money in and money out of similar large size in the same month): the money
  passed through the account instead of staying in it, which can be the layering stage of money laundering; staff
  should trace where the money came from and where it went.
- Burst (burst share 25% or more): repeated transactions with the same counterparty in one month; staff should
  identify that counterparty and the reason.
- Activity after inactivity: the account was quiet and then moved money; staff should confirm who used it and why.
- Long gaps are context: they help staff ask the right questions. A long gap is not a burst.

Silently work through these steps:
1. Read RISK_SIGNALS and each CHECK line. pattern_found checks are the points to write about. insight=none means no
   explanation is available, so use that check's months in the MONTH rows instead. insight_issues lists where that
   insight or its pattern does not match the CSV; when it is not none, trust the MONTH rows over that insight.
2. Connect the findings rather than repeating them, for example an unusual transaction together with layering
   months, a burst month or a quiet period before it.
3. Do not describe debit flow, credit flow, debit-only, credit-only or mixed months, or rises and falls in debits or
   credits; they are not findings.
4. Judge whether the evidence fits the customer's declared occupation and individual or organisation type: that is
   your occupation_fit.
5. Write "money laundering" only when unusual_transaction or layering_months lists a month; otherwise describe the
   points as context for staff to check.
6. Decide the risk only with the RISK GUIDE at the end.

Write:
headline: one line, at most 120 characters, naming the most important finding about this account; it must agree with
point_1.
point_1, point_2, point_3: each one plain sentence, at most 220 characters: a finding with its month and amount, and
what it means. Most important first: layering and an unusual transaction come before burst, inactivity or gaps. Use
the JSON string "none" for point_2 or point_3 when there is nothing more of value; never repeat a point. Never write
field names such as layering_months=none in any line.
why_it_matters: one or two sentences, at most 260 characters, explaining in plain AML terms why these findings together
matter for due diligence.
occupation_fit: one or two sentences, at most 260 characters: whether the unusual transactions and layering months
(or, when there are none, the account's usual amounts) fit the declared occupation and type, and why. Everyday amounts
fit any occupation. With PROFILE|none supplied, write that no customer profile was available to compare.
verify_1, verify_2: each one concrete action for staff, at most 200 characters, tied to a specific month and amount
(for example: ask where the money in a layering month came from and where it went). Use "none" for verify_2 when
one action is enough.
risk_reason: one sentence, at most 200 characters. Start with what RISK_SIGNALS shows: the unusual transaction (or
that there is none) and the layering months (or that there is no layering).
risk_level: exactly "low", "medium" or "high", decided as the RISK GUIDE at the end teaches.

STRICT JSON ONLY. Return one RFC 8259 JSON object. Double-quote every key and string. Do not use markdown, arrays,
nested objects, or extra keys. Write the keys in exactly this order:
"headline", "point_1", "point_2", "point_3", "why_it_matters", "occupation_fit", "verify_1", "verify_2",
"risk_reason", "risk_level".
Write the object once; do not repeat it or explain your steps. After the final } output no other character.

RISK GUIDE: decide risk_level from the RISK_SIGNALS line only. Check in this order and use the first that applies:
1. "high" when unusual_transaction lists a month and layering_months lists a month (they can be different months).
2. "medium" when unusual_transaction lists a month and layering_months=none. A burst month or activity after
   inactivity keeps it "medium".
3. "low" when unusual_transaction=none. It stays "low" even when layering_months lists a month, and with bursts,
   long gaps, inactivity or an occupation question; write those as points for staff to verify.
The screening has already measured the amounts against this account's usual level; do not judge them again.
occupation_fit, long gaps, quiet months and NOT_VERIFIED never change risk_level.
"""


PROFILE_CONTEXT_SYSTEM_PROMPT = """You assist authorised bank staff with customer-profile context.

Use only INPUT FACTS: the customer's dated profile versions and six monthly transaction aggregates. Describe the
profile as evidence; do not judge whether the activity fits the occupation and do not recommend checks: a later step
does that. State the declared occupation and individual/organisation type when supplied. Cite the largest single
transaction and its month, and how it compares with the account's usual largest when PROFILE_ACTIVITY_FOCUS gives one;
when it says not_enough_history, say there is not enough history to compare. A monthly total is transaction volume, not income or net funds received. Occupation does not prove income,
wealth, or that a transaction is unsuitable; do not label a job low-income.
An individual may legitimately transact large amounts, and an organisation may transact small amounts.

State the supplied citizenship as a profile fact. The six monthly rows contain no transaction-country, residency,
counterparty-location, or source-of-funds field. Therefore citizenship cannot establish where the money originated,
whether any transaction was cross-border, or a country-risk rating; citizenship is not where the activity happened.
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


def activity_money_prompt(monthly_summary: list[dict[str, Any]], screening: Screening) -> str:
    """Call 1 instructions: the fixed prompt, then the screening's two findings, last."""
    return (ACTIVITY_MONEY_SYSTEM_PROMPT + unusual_transaction_note(monthly_summary, screening)
            + layering_note(monthly_summary, screening))


def activity_money_input(monthly_summary: list[dict[str, Any]]) -> str:
    """Call 1 input: the account's usual level, then the figures each note refers to. No raw rows:
    the labelled lines hold every figure the descriptions need, with less to misread."""
    usual = account_usual(monthly_summary)
    return "\n".join((
        "INPUT FACTS — amounts in RM, months oldest to newest.",
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