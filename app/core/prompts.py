from __future__ import annotations

from typing import Any


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
2. Compare the highest and lowest txn_count_monthly and total_amount months, then check avg_amount, max_amount and
   std_amount. Describe a material increase, decrease, or concentration accurately; do not call the first supplied
   month historical baseline.
3. Compare debit_count_monthly, credit_count_monthly, monthly_debit and monthly_credit by month. State only the
   direction actually shown. A change between debit and credit activity is context, not an allegation.
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

Each outcome is exactly observed or not_observed. Each *_months value contains exactly two different supplied months
as YYYYMM,YYYYMM with no spaces; it is a comparison pair, not an evidence identifier. Never output none, N/A, M,
field names, or an underscore in a *_months value. Each context must mention only what the selected months and their
rows show. The executive summary must not say reactivation or dormancy unless dormancy_outcome is observed. After the
final } output no other character.
"""


PROFILE_CONTEXT_SYSTEM_PROMPT = """You are a customer-profile summarisation assistant for authorised bank staff.

Summarise only factual customer-profile fields supplied in INPUT FACTS. Do not assess transaction risk, profile
consistency, customer suitability, AML, or wrongdoing. Do not infer occupation duties, income, account purpose, source
of funds, geography, or any missing field. Do not ask the caller for more information. Citizenship may be stated only
as a factual profile attribute and must never be connected to risk.

Use only fields that have an explicit value. Do not output N/A, insufficient_data, missing, unavailable, not provided,
or a statement about data that is not supplied. Where present, occupation_name is the resolved occupation label and
must be preferred over occupation_code. Dates are factual profile timeline dates only.

STRICT JSON ONLY. Return one RFC 8259 JSON object with exactly one key: "profile_summary". The value is one factual
sentence under 220 characters. Do not refer to P1, P2, record IDs, labels, brackets, or a customer name. Include the
resolved occupation name when present; state citizenship only as a factual attribute. After the final } output no other
character.
"""


FORMAT_RETRY_SUFFIX = """

FORMAT RETRY: Redo the task using INPUT FACTS. Return only the required JSON object. Do not output explanation,
questions, assistant, system, user, analysis, markdown, or text before or after the final }.
"""


def transaction_context_input(monthly_summary: list[dict[str, Any]]) -> str:
    rows = ["|".join(_TRANSACTION_FIELDS)]
    rows.extend(
        "|".join(str(row.get(field, "")) for field in _TRANSACTION_FIELDS)
        for row in monthly_summary
    )
    return "INPUT FACTS — six monthly rows, oldest to newest. Do not copy them into the response.\n" + "\n".join(rows)


def profile_context_input(customer_profile: list[dict[str, str | None]]) -> str:
    """Render only supplied profile facts in a compact, model-readable form."""
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
    return "\n".join(lines)
