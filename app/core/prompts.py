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

Silently complete this lookup procedure before writing the answer:
1. Read txn_count_monthly in all six rows for dormancy. Reactivation requires three immediately preceding zero-count
   months followed by activity.
2. Compare the highest and lowest supplied txn_count_monthly and total_amount months. Then check avg_amount,
   max_amount, and std_amount before describing activity/value change. The first supplied month is not earlier history.
3. Read debit_count_monthly, credit_count_monthly, monthly_debit, and monthly_credit by month before describing a
   debit/credit pattern. Use debit and credit labels exactly; do not invent flow direction.
4. Read pct_burst and pct_trx_gap by month. pct_burst of zero cannot support burst activity. Without a supplied
   business definition, pct_trx_gap is only a pattern, not a suspicious act.

Each of the four checks is assessable from these rows. Use observed only for a material pattern that warrants staff
context; otherwise use not_observed and still state the actual pattern. Do not output N/A, insufficient_data, a
generic "nothing happened" statement, or a request for more information.

Risk policy: low requires no observed transaction check; medium requires at least one observed check; high requires at
least two observed checks. High is never based on amount alone. Keep the summary under 260 characters and each context
under 140 characters.

STRICT JSON ONLY. Return one RFC 8259 JSON object. Double-quote every key and string. Do not use markdown, prose,
examples, placeholders, arrays, nested objects, task keys, or extra keys. The object must contain exactly these keys:
"risk_level", "executive_summary",
"dormancy_outcome", "dormancy_context", "dormancy_evidence",
"activity_value_outcome", "activity_value_context", "activity_value_evidence",
"debit_credit_outcome", "debit_credit_context", "debit_credit_evidence",
"burst_gap_outcome", "burst_gap_context", "burst_gap_evidence".

Each outcome is exactly observed or not_observed. Each evidence field contains exactly two comma-separated monthly IDs
from different months, with no spaces. Form an ID as MYYYYMM.field_name, using only a field in the supplied rows.
After the final } output no other character.
"""


PROFILE_CONTEXT_SYSTEM_PROMPT = """You are a customer-profile summarisation assistant for authorised bank staff.

Summarise only factual customer-profile fields supplied in INPUT FACTS. Do not assess transaction risk, profile
consistency, customer suitability, AML, or wrongdoing. Do not infer occupation duties, income, account purpose, source
of funds, geography, or any missing field. Do not ask the caller for more information. Citizenship may be stated only
as a factual profile attribute and must never be connected to risk.

Use only fields that have an explicit value. Do not output N/A, insufficient_data, missing, unavailable, not provided,
or a statement about data that is not supplied. Where present, occupation_name is the resolved occupation label and
must be preferred over occupation_code. Dates are factual profile timeline dates only.

STRICT JSON ONLY. Return one RFC 8259 JSON object. Double-quote every key and string. Do not use markdown, prose,
examples, placeholders, arrays, nested objects, task keys, or extra keys. The object must contain exactly these keys:
"profile_summary", "profile_evidence". profile_summary is one factual sentence under 220 characters.
profile_evidence contains exactly two comma-separated P profile IDs with no spaces, selected only from fields present
in INPUT FACTS. Each supplied field shows its P evidence ID in square brackets; cite that exact ID. After the final }
output no other character.
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
    for record in customer_profile:
        values = [f"record_id={record['profile_record_id']}"]
        values.extend(
            f"{label}[{record['profile_record_id']}.{field}]={record[field]}"
            for field, label in field_labels
            if record.get(field) is not None
        )
        lines.append("PROFILE_RECORD|" + "|".join(values))
    return "\n".join(lines)
