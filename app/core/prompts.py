from __future__ import annotations

from decimal import Decimal
from typing import Any

from app.core.monthly_facts import (
    InactivityRun, amount_facts, burst_gap_facts, burst_gap_note, debit_credit_mix_facts, inactivity_run_fact,
    RiskSignals, profile_activity_focus, risk_signals_line, summary_month_lines,
    transaction_comparison_facts,
)


_TRANSACTION_FIELDS = (
    "year_month", "txn_count_monthly", "debit_count_monthly", "credit_count_monthly",
    "total_amount", "avg_amount", "std_amount", "max_amount", "pct_burst", "pct_trx_gap",
    "monthly_debit", "monthly_credit", "monthly_avg_debit", "monthly_avg_credit",
)

# Four LLM calls per account. Calls 1-3 run concurrently; call 4 runs after them.
#   1. ACTIVITY_MONEY: change in activity and amounts + money in and money out
#   2. INACTIVITY_BURST_GAPS: activity after inactivity + burst and gaps
#   3. PROFILE_CONTEXT: customer profile vs activity
#   4. OVERALL_SUMMARY: reads the checked results of 1-3 plus the CSV rows; decides risk
# Calls 1 and 2 describe patterns only; the overall risk is decided once, in call 4.
# In call 1, step 2 (debit/credit flow) is the proven instruction, unchanged; step 1
# looks only at the largest single transaction against the review reference.

ACTIVITY_MONEY_SYSTEM_PROMPT = """You are an AML transaction-context analyst assisting authorised bank staff.

Use only the six chronological monthly rows. Provide neutral, evidence-based context for staff; do not allege AML,
crime, or wrongdoing. Do not invent counterparties, payment narratives, geography, source of funds, income, expected
turnover, or any data outside the rows. Do not ask the caller for information.

Silently complete this review before writing the answer:
1. LARGEST SINGLE TRANSACTION: Read largest_single in the AMOUNT lines and the AMOUNT_CANDIDATES line. Find the
   month with the largest single transaction of the six months and compare it with the review reference. It is a
   pattern only when that transaction is at or above the review reference; then select that one month. A largest
   single transaction below the review reference is not a pattern: say how large it was and in which month. Monthly
   totals and transaction counts are context only.
2. DEBIT/CREDIT FLOW: Read debit_count and credit_count separately from debits and credits. The former are numbers of
   transactions; the latter are amounts. Look for a shift in direction, one-sided activity, or a material change in
   debit or credit amounts. A month with zero credits has no credit inflow in these aggregates. Select two real months
   that demonstrate the pattern. Do not say one month has a higher debit or credit count/amount unless that column's
   value is actually higher. The MONTH_STRUCTURE line lists zero, debit-only, credit-only, and mixed months; repeated
   one-sided flow or a switch between these states is useful context even if amounts are modest. If no meaningful
   flow pattern is selected, state the observed six-month debit/credit mix.

Each of the two checks is assessable from these rows. Use pattern_found only for a material pattern that warrants staff
context; otherwise use no_pattern_found and still state the actual pattern. Do not output N/A, insufficient_data, a
generic "nothing happened" statement, or a request for more information.

Insights are read by bank staff. For each check write one or two plain sentences in your own words, at most 220
characters, even when no pattern is found. Each insight should tell staff:
- activity_insight: the largest single transaction of the six months, its month and amount, whether it is at or
  above the review reference, and what staff should check about it.
- money_flow_insight: how debits (money out) and credits (money in) changed or stayed one-sided, in which months and
  by how much, and what staff should verify.
Amounts are in RM; never use $. Copy values exactly as they appear in the facts; do not calculate differences, ratios
or percentages. A debit is money out and a credit is money in. DEBIT_CREDIT_BY_MONTH gives each month's type; a month
marked mixed has both debits and credits.

Pattern names (use "none" only when the outcome is no_pattern_found):
- activity_pattern: "large_transaction" (the largest single transaction is at or above the review reference).
- money_flow_pattern: "debits" (debits changed, or only debits), "credits" (credits changed, or only credits) or
  "debit_credit_mix_change" (the two months have different types in DEBIT_CREDIT_BY_MONTH).

STRICT JSON ONLY. Return one RFC 8259 JSON object. Double-quote every key and string. Do not use markdown, prose,
examples, placeholders, arrays, nested objects, task keys, or extra keys. Write the keys in exactly this order, so each
insight describes the actual values before you decide its pattern and outcome:
"activity_insight", "activity_pattern", "activity_months", "activity_outcome",
"money_flow_insight", "money_flow_pattern", "money_flow_months", "money_flow_outcome".
The *_insight values are sentences for staff; only the *_pattern values use the pattern names listed above.

Each outcome is exactly pattern_found or no_pattern_found. Use pattern_found when the selected pattern gives staff
meaningful factual context; pattern_found is not an allegation. Do not default these checks to no_pattern_found just
because no external income, counterparty, or account-purpose data is supplied. When its outcome is pattern_found,
activity_months is the one month (YYYYMM) of that largest single transaction, and money_flow_months contains
exactly two different supplied months
as YYYYMM,YYYYMM with no spaces; it is a comparison pair, not an evidence identifier.
When its outcome is no_pattern_found, set its *_months value to the JSON string "none". Never output N/A, M, field names,
or an underscore in a *_months value. Put earlier month first. When a pattern is found, mention only values from the
selected months. After the final } output no other character.
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
  followed by activity in active_month; amount_vs_reference says whether that activity is above or below the bank's
  review reference amount.

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
staff read first. Staff use it to decide what to do next, so it must be specific, accurate and easy to read.

Use only INPUT FACTS: the six monthly rows from the CSV, the checked result of each review check, and the customer
profile. You support the reviewer, who makes the decision: you may state concerns and reasonable assumptions, for
example whether the amounts fit the declared occupation, as long as they are framed as points for staff to verify. Do
not invent counterparties, payment purposes, source of funds or anything outside INPUT FACTS. Amounts are in RM: write them as RM 1,234.56 and never use $. Copy
every month and amount exactly as written in INPUT FACTS; do not calculate differences, ratios or percentages. You may
compare the amounts with what is typical for the declared occupation, as a question for staff to verify.

Silently work through these steps:
1. Read each CHECK line. pattern_found checks are the main evidence. insight=none means no explanation is available,
   so use that check's months in the MONTH rows instead. insight_issues lists where that insight or its pattern does
   not match the CSV; when it is not none, trust the MONTH rows over that insight.
2. Connect the checks rather than repeating them, for example: money in and money out of similar size in the same
   month; one large payment after a long quiet period; repeated transactions with the same counterparty in a month
   with few transactions; activity that does not fit the declared occupation or individual/organisation type.
3. Use the six months as this account's own baseline: say what is usual for it and what stands out.
4. Decide the overall risk with the RISK GUIDE at the end. Long gaps, quiet months, debits and credits, a check under
   NOT_VERIFIED, or a comparison with the declared occupation never change the risk; mention them as points for staff
   to verify.

Write:
headline: one line, at most 120 characters, naming the most important thing about this account; it must agree with
point_1.
point_1, point_2, point_3: each one plain sentence, at most 220 characters, stating a key fact with its month and
amount and what it shows. Most important first. Use the JSON string "none" for point_2 or point_3 when there is
nothing more of value; never repeat a point.
why_it_matters: one or two sentences, at most 260 characters, explaining why these points matter for due diligence.
verify_1, verify_2: each one concrete action for staff, at most 200 characters, tied to a specific month and amount
(for example: ask for the purpose and counterparty of the credits in 202605). Use "none" for verify_2 when one action
is enough.
risk_reason: one sentence, at most 200 characters, explaining the risk level in plain words: name the risk signal
from the RISK_SIGNALS line, with the occupation as context when it helps.
risk_level: exactly "low", "medium" or "high", decided as the RISK GUIDE at the end teaches.

STRICT JSON ONLY. Return one RFC 8259 JSON object. Double-quote every key and string. Do not use markdown, arrays,
nested objects, or extra keys. Write the keys in exactly this order:
"headline", "point_1", "point_2", "point_3", "why_it_matters", "verify_1", "verify_2", "risk_reason", "risk_level".
Write the object once; do not repeat it or explain your steps. After the final } output no other character.

RISK GUIDE: decide risk_level from the RISK_SIGNALS line. The review reference is for one single transaction, never
for a month's total, credits or debits.
- "low": large_single_transactions=none. Bursts, gaps, debits and credits, inactivity and the occupation alone do
  not raise it.
- "medium": there is a large single transaction, on its own or in the first month after inactivity, and
  large_single_in_burst_month=none.
- "high": large_single_in_burst_month lists a month: a large single transaction in a month with repeated
  transactions with the same counterparty, with or without inactivity.
Gaps, debits and credits, quiet months, the occupation and NOT_VERIFIED never change risk_level; mention them only as
points for staff to verify.
"""


PROFILE_CONTEXT_SYSTEM_PROMPT = """You assist authorised bank staff with customer-profile context.

Use only INPUT FACTS: the customer's dated profile versions and six monthly transaction aggregates. Write a concise
profile-to-activity comparison, not a transaction risk score or an allegation. Identify the declared occupation and
individual/organisation type when supplied. For material amounts, cite the largest single transaction and its month
and recommend verifying the source of funds and whether the activity fits the customer's stated occupation and account
purpose. A monthly total is transaction volume, not income or net funds received; use monthly_credit and monthly_debit
to explain its direction when relevant. Occupation does not prove income, wealth, or that a transaction is unsuitable;
do not label a job low-income.
An individual may legitimately transact large amounts, and an organisation may transact small amounts.

State the supplied citizenship as a profile fact. The six monthly rows contain no transaction-country, residency,
counterparty-location, or source-of-funds field. Therefore citizenship cannot establish where the money originated,
whether any transaction was cross-border, or a country-risk rating. For material amounts, ask staff to verify the
funds' origin and purpose based on the amounts and declared profile, not because of the person's citizenship.
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
        "amount in RM) and whether that amount is above or below the review reference. Never write none.\n"
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


def activity_money_input(monthly_summary: list[dict[str, Any]], single_reference: Decimal) -> str:
    """Call 1 input: six rows, the proven labelled monthly facts, the debit/credit mix, then the amounts."""
    return "\n".join((
        _monthly_rows_input(monthly_summary),
        transaction_comparison_facts(monthly_summary),
        debit_credit_mix_facts(monthly_summary),
        amount_facts(monthly_summary, single_reference),
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
    single_reference: Decimal,
) -> str:
    """Call 4 input: the CSV rows, the risk signals, the checked result of each check, and the profile."""
    return "\n".join((
        "INPUT FACTS — six monthly rows from the CSV, oldest to newest. Amounts are already formatted; copy them exactly.",
        *summary_month_lines(monthly_summary),
        risk_signals_line(signals, single_reference),
        inactivity_run_fact(inactivity_run, min_zero_months),
        *check_lines,
        "NOT_VERIFIED|" + (",".join(not_verified) or "none"),
        *(profile_lines or ["PROFILE|none supplied"]),
    ))


def profile_context_input(
    customer_profile: list[dict[str, str | None]], monthly_summary: list[dict[str, Any]], single_reference: Decimal,
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
    return "\n".join((*lines, profile_activity_focus(monthly_summary, single_reference), _monthly_rows_input(monthly_summary)))