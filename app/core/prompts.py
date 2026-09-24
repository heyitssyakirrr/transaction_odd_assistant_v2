from __future__ import annotations

from typing import Any


ANALYST_SYSTEM_PROMPT = """You are an AML transaction-review analyst assisting bank staff.
Use only the supplied monthly account summary and customer profile history.
Do not infer or invent customer occupation, income, source of wealth, intent, criminal conduct, external risk data,
regulations, or facts absent from the supplied data. An unusual pattern is a review indicator, not proof of money
laundering.

Every material finding must cite the year_month and the specific feature/column name that supports it, using only
the monthly rows supplied. Do not cite a month or feature that was not supplied.
Do not report every month: report material activity only. Do not report routine, expected month-to-month variation
as a finding; an empty findings list is valid. Keep every text field brief.
Return JSON only and follow the requested schema exactly.
Return the object described by response_schema directly; never wrap it in a "response_schema" key.

Output format: respond with a single JSON object as compact, single-line JSON —
no line breaks, indentation, or extra whitespace inside it. Do not wrap it in
markdown or code fences. Output nothing before or after the JSON object: no
greeting, no explanation, no closing remark. Stop immediately after the final
closing brace."""


_EVIDENCE_ITEM_SCHEMA: dict[str, Any] = {
    "year_month": "YYYYMM (e.g. 202604), must exactly match one of the supplied monthly rows",    "feature": "column name this finding is based on, e.g. pct_burst",
    "value": "that feature's value for that month, as supplied",
    "statement": "string, maximum 20 words",
}

_FINDING_SCHEMA: dict[str, Any] = {
    "category": "string, maximum 5 words",
    "severity": "low|medium|high|critical",
    "evidence": [_EVIDENCE_ITEM_SCHEMA],
    "rationale": "string, maximum 30 words",
}


def account_assessment_payload(
    monthly_summary: list[dict[str, Any]],
    customer_profile: list[dict[str, Any]],
) -> dict[str, Any]:
    """Ask the model to assess AML risk for one account from its 6-month
    summary and any customer_info history overlapping that window."""
    task = (
        "Review this account's 6-month monthly summary and its linked customer profile history as one case. "
        "Compare each month against the account's own 6-month pattern -- do this reasoning yourself, no "
        "precomputed anomaly scores are supplied. Specifically check for: month-over-month spikes in "
        "total_amount or txn_count_monthly; rising pct_burst (bursty/clustered transaction timing); irregular "
        "day_gaps or pct_trx_gap (e.g. dormancy followed by sudden activity); shifts in debit/credit balance "
        "(monthly_debit vs monthly_credit, debit_count_monthly vs credit_count_monthly), e.g. flipping from "
        "credit-heavy to debit-heavy, or high matched turnover suggesting pass-through/mule-like behaviour; a "
        "high max_amount vs avg_amount ratio, i.e. one outlier transaction dominating a month; and any "
        "customer_profile change (occupation, citizenship) that coincides with a transaction-pattern change in "
        "the same window. Use occupation_cd only as a coarse plausibility check against transaction "
        "volume/velocity, never as a determination. Do not report routine, expected activity as a finding. "
        "Return at most 5 findings and at most 3 limitations."
    )
    response_schema: dict[str, Any] = {
        "risk_level": "low|medium|high",
        "executive_summary": "string, maximum 120 words",
        "findings": [_FINDING_SCHEMA],
        "limitations": ["string, maximum 20 words"],
    }
    return {
        "task": task,
        "monthly_summary": monthly_summary,
        "customer_profile": customer_profile,
        "response_schema": response_schema,
    }