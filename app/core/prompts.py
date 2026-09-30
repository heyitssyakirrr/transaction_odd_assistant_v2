from __future__ import annotations

from decimal import Decimal
from typing import Any

from app.core.monthly_facts import transaction_comparison_facts


_TRANSACTION_FIELDS = (
    "year_month", "txn_count_monthly", "debit_count_monthly", "credit_count_monthly",
    "total_amount", "avg_amount", "std_amount", "max_amount", "pct_burst", "pct_trx_gap",
    "monthly_debit", "monthly_credit", "monthly_avg_debit", "monthly_avg_credit",
)

TRANSACTION_CONTEXT_SYSTEM_PROMPT = """You are an AML transaction-context analyst assisting authorised bank staff.

Use only the six chronological monthly rows. Provide neutral, evidence-based context for staff; do not allege AML,
crime, or wrongdoing. Do not invent counterparties, payment narratives, geography, source of funds, income, expected
turnover, or any data outside the rows. Do not ask the caller for information.

Silently complete this review before writing the answer:
1. Read txn_count_monthly across all six rows. Reactivation requires three immediately preceding zero-count months
   followed by activity. Do not call a quiet, declining, or fluctuating account dormant.
2. ACTIVITY/VALUE: Read the six labelled MONTH facts and the two ACTIVITY_CANDIDATES. Compare counts with counts and
   monthly totals with monthly totals. A count change and a value change can occur in different month pairs. Select
   the two months that best show a meaningful increase, decrease, zero-to-active change, or concentration. State the
   exact values for those two months only; never interpolate an intermediate month or call the first month a baseline.
   If no meaningful pattern is selected, say what the six-month counts and totals actually show.
3. DEBIT/CREDIT FLOW: Read debit_count and credit_count separately from debits and credits. The former are numbers of
   transactions; the latter are amounts. Look for a shift in direction, one-sided activity, or a material change in
   debit or credit amounts. A month with zero credits has no credit inflow in these aggregates. Select two real months
   that demonstrate the pattern. Do not say one month has a higher debit or credit count/amount unless that column's
   value is actually higher. The MONTH_STRUCTURE line lists zero, debit-only, credit-only, and mixed months; repeated
   one-sided flow or a switch between these states is useful context even if amounts are modest. If no meaningful
   flow pattern is selected, state the observed six-month debit/credit mix.
4. Read pct_burst and pct_trx_gap. A pct_burst of zero cannot support burst activity. pct_trx_gap is a pattern only;
   no business definition or suspicious act may be inferred from it.

Each of the four checks is assessable from these rows. Use observed only for a material pattern that warrants staff
context; otherwise use not_observed and still state the actual pattern. Do not output N/A, insufficient_data, a
generic "nothing happened" statement, or a request for more information.

Risk policy: low requires no observed transaction check; medium requires at least one observed check; high requires at
least two observed checks. High is never based on amount alone. Keep the summary under 260 characters and each context
under 140 characters.

STRICT JSON ONLY. Return one RFC 8259 JSON object. Double-quote every key and string. Do not use markdown, prose,
examples, placeholders, arrays, nested objects, task keys, or extra keys. The object must contain exactly these keys:
"risk_level", "executive_summary",
"dormancy_outcome", "dormancy_context", "dormancy_months",
"activity_value_outcome", "activity_value_context", "activity_value_months",
"debit_credit_outcome", "debit_credit_context", "debit_credit_months",
"burst_gap_outcome", "burst_gap_context", "burst_gap_months".

Each outcome is exactly observed or not_observed. For activity_value and debit_credit, use observed when the selected
pattern gives staff meaningful factual context; observed is not an allegation. Do not default these checks to
not_observed just because no external income, counterparty, or account-purpose data is supplied. Each *_months value
contains exactly two different supplied months
as YYYYMM,YYYYMM with no spaces when its outcome is observed; it is a comparison pair, not an evidence identifier.
When its outcome is not_observed, set its *_months value to the JSON string "none". Never output N/A, M, field names, or an
underscore in a *_months value. Put earlier month first. Use YYYYMM rather than month names in the context; when
observed, mention only values from the selected two months. Do not write a numerical claim in the executive summary
unless it matches a selected pair. The executive summary must not say reactivation or dormancy unless
dormancy_outcome is observed. After the final } output
no other character.
"""


PROFILE_CONTEXT_SYSTEM_PROMPT = """You assist authorised bank staff with customer-profile context.

Use only INPUT FACTS: the customer's dated profile versions and six monthly transaction aggregates. Write a concise
profile-to-activity comparison, not a transaction risk score or an allegation. Identify the declared occupation and
individual/organisation type when supplied. For material amounts, cite a real month and its total_amount or max_amount
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
No markdown, arrays, extra keys, record IDs, or text after the final }.
"""


FORMAT_RETRY_SUFFIX = """

FORMAT RETRY: Redo the task using INPUT FACTS. Return only the required JSON object. Do not output explanation,
questions, assistant, system, user, analysis, markdown, or text before or after the final }.
"""


def _monthly_rows_input(monthly_summary: list[dict[str, Any]]) -> str:
    rows = ["|".join(_TRANSACTION_FIELDS)]
    rows.extend(
        "|".join(str(row.get(field, "")) for field in _TRANSACTION_FIELDS)
        for row in monthly_summary
    )
    return "INPUT FACTS — six monthly rows, oldest to newest. Do not copy them into the response.\n" + "\n".join(rows)


def transaction_context_input(monthly_summary: list[dict[str, Any]]) -> str:
    return _monthly_rows_input(monthly_summary) + "\n" + transaction_comparison_facts(monthly_summary)


def profile_context_input(customer_profile: list[dict[str, str | None]], monthly_summary: list[dict[str, Any]]) -> str:
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
    peak_total = max(monthly_summary, key=lambda row: Decimal(str(row["total_amount"])))
    peak_single = max(monthly_summary, key=lambda row: Decimal(str(row["max_amount"])))
    focus = (
        "PROFILE_ACTIVITY_FOCUS|"
        f"highest_monthly_total={peak_total['year_month']}:{Decimal(str(peak_total['total_amount'])):.2f}|"
        f"same_month_debit={Decimal(str(peak_total['monthly_debit'])):.2f}|"
        f"same_month_credit={Decimal(str(peak_total['monthly_credit'])):.2f}|"
        f"largest_reported_single_amount={peak_single['year_month']}:{Decimal(str(peak_single['max_amount'])):.2f}"
    )
    return "\n".join(lines) + "\n" + focus + "\n" + _monthly_rows_input(monthly_summary)
