from __future__ import annotations

import json
from typing import Any


ANALYST_SYSTEM_PROMPT_TEMPLATE = """You are an AML transaction-review analyst assisting bank staff.
Use only the supplied monthly account summary and customer profile history.
Do not infer or invent customer income, source of wealth, intent, criminal conduct, external risk data,
regulations, or facts absent from the supplied data. An unusual pattern is a review indicator, not proof of money
laundering.

You are not given any precomputed anomaly scores, deltas, or statistics. Do that comparison yourself, in plain
language, as the monthly_comparison step below -- do not silently estimate numbers you were not given.

Work in this order, and only this order, because each step depends on the one before it:
1. monthly_comparison -- for each feature worth discussing, compare its 6 monthly values against each other for
   THIS account only (this account's own 6 months are its baseline, not any fixed threshold). Note which
   year_month(s) stand out and how (e.g. "flat near zero for 5 months, then one nonzero month" or "steadily
   rising each month" or "one month far above the rest"). Skip features that look routine; do not force a note
   for every feature. An empty list is valid.
2. profile_timeline_notes -- note any occupation, citizenship, or indv_org_type change in the supplied
   customer_profile history: when it took effect, and whether that timing overlaps a month flagged in
   monthly_comparison. The occupation or citizenship VALUE itself is never a risk factor and must never appear
   as a reason in any rationale -- only the TIMING of a change relative to a transaction-pattern change is
   relevant. An empty list is valid.
3. findings -- turn the notes above into findings, citing only year_month/feature values that were actually
   supplied. If multiple features change together in the same month as part of one underlying event (e.g. an
   account reactivating after months of dormancy), report that as ONE finding with multiple evidence entries --
   never split one event into several findings. Severity depends on (a) how far the flagged month(s) are from
   this account's own pattern in its other months, and (b) how many independent features or profile-timing notes
   corroborate the same month: more independent corroboration means higher severity. Three or more consecutive
   zero-activity months followed directly by any activity is at least medium severity regardless of the
   transaction amount involved -- the dormancy-to-reactivation shape is itself the signal, not the dollar
   figure. Do not report routine, expected month-to-month variation as a finding; an empty findings list is
   valid.

Every finding's evidence must cite only year_month and feature values from the supplied monthly rows -- never a
month or feature that was not supplied. Keep every text field brief. Return at most 8 monthly_comparison notes,
5 findings, and 3 limitations.

Return JSON only and follow the requested schema exactly, in the field order given.
Return the object described by response_schema directly; never wrap it in a "response_schema" key.

Output format: respond with a single JSON object as compact, single-line JSON --
no line breaks, indentation, or extra whitespace inside it. Do not wrap it in
markdown or code fences. Output nothing before or after the JSON object: no
greeting, no explanation, no closing remark. Stop immediately after the final
closing brace.

Every entry in monthly_comparison, profile_timeline_notes, and findings MUST be a JSON object with exactly the
keys shown in response_schema -- never a plain string, never a "key: value" string, never "category: severity".
If you have nothing to report for a step, return an empty array [] for that step; do not shrink an object down to
a string summary instead.

{examples_block}"""


# Few-shot examples shown to the model as part of the system prompt, below.
# These are deliberately fabricated, self-consistent mini-cases -- NOT
# derived from the schema description dicts above -- so the model sees a
# literal, complete, correctly-typed response object rather than another
# layer of field descriptions. Rendered with json.dumps(..., separators=
# (",", ":")) to match the exact "compact single-line JSON" output format
# we require, so the example doubles as a formatting demonstration.
#
# Two examples on purpose:
#   - EXAMPLE_WITH_FINDING shows the *shape* of a populated finding,
#     including multi-evidence grounding for one underlying event (per the
#     "never split one event into several findings" rule) and a
#     profile_timeline_note that coincides with it.
#   - EXAMPLE_QUIET shows that empty arrays are a valid, complete response
#     on their own -- so the model doesn't feel pressure to manufacture a
#     finding just to have "something" in every array.
_EXAMPLE_WITH_FINDING: dict[str, Any] = {
    "monthly_comparison": [
        {
            "feature": "txn_count_monthly",
            "pattern_summary": "Zero or near-zero for 5 months, then a sharp jump in the most recent month.",
            "notable_months": ["202607"],
        },
        {
            "feature": "monthly_credit",
            "pattern_summary": "Flat near zero for 5 months, then a large single-month credit inflow.",
            "notable_months": ["202607"],
        },
    ],
    "profile_timeline_notes": [
        {
            "field": "occupation",
            "change_summary": "Occupation code changed partway through the review window.",
            "change_dttm": "2026-07-02T00:00:00",
            "coincides_with_txn_pattern": True,
        }
    ],
    "findings": [
        {
            "category": "reactivation",
            "severity": "medium",
            "evidence": [
                {
                    "year_month": "202607",
                    "feature": "txn_count_monthly",
                    "value": "42",
                    "statement": "Transaction count jumped from ~0 in prior months to 42 this month.",
                },
                {
                    "year_month": "202607",
                    "feature": "monthly_credit",
                    "value": "18500.00",
                    "statement": "Credit inflow jumped from ~0 in prior months to 18,500 this month.",
                },
            ],
            "rationale": (
                "Five months of near-zero activity followed by a same-month jump in both transaction "
                "count and credit inflow, coinciding with a profile change."
            ),
        }
    ],
    "executive_summary": (
        "The account was dormant for five months and then reactivated sharply in the sixth month, with "
        "both transaction volume and credit inflow rising together. This coincided with an occupation "
        "change on record. No other features showed notable deviation."
    ),
    "limitations": ["Assessment is limited to the 6 supplied monthly rows and profile history."],
}

_EXAMPLE_QUIET: dict[str, Any] = {
    "monthly_comparison": [
        {
            "feature": "avg_amount",
            "pattern_summary": "Gradually rising each month, consistent with the account's own trend.",
            "notable_months": [],
        }
    ],
    "profile_timeline_notes": [],
    "findings": [],
    "executive_summary": (
        "Monthly activity is broadly consistent across the 6-month window, with only a routine gradual "
        "rise in average transaction amount. No profile changes were supplied. No findings are reported."
    ),
    "limitations": ["Assessment is limited to the 6 supplied monthly rows and profile history."],
}


def _render_example(example: dict[str, Any]) -> str:
    return json.dumps(example, separators=(",", ":"), default=str)


_EXAMPLES_BLOCK = (
    "Example 1 -- a case with a finding (fabricated data, structure only, do not copy any content):\n"
    f"{_render_example(_EXAMPLE_WITH_FINDING)}\n\n"
    "Example 2 -- a quiet case with no findings (empty arrays are a complete, valid response):\n"
    f"{_render_example(_EXAMPLE_QUIET)}"
)

ANALYST_SYSTEM_PROMPT = ANALYST_SYSTEM_PROMPT_TEMPLATE.format(examples_block=_EXAMPLES_BLOCK)


_EVIDENCE_ITEM_SCHEMA: dict[str, Any] = {
    "year_month": "YYYYMM (e.g. 202604), must exactly match one of the supplied monthly rows",
    "feature": "column name this finding is based on, e.g. pct_burst",
    "value": "that feature's value for that month, as supplied",
    "statement": "string, maximum 20 words",
}

_FINDING_SCHEMA: dict[str, Any] = {
    "category": "string, maximum 5 words, e.g. reactivation, amount_spike, flow_reversal, pass_through",
    "severity": "low|medium|high|critical",
    "evidence": [_EVIDENCE_ITEM_SCHEMA],
    "rationale": "string, maximum 30 words",
}

_MONTHLY_COMPARISON_SCHEMA: dict[str, Any] = {
    "feature": "column name, e.g. txn_count_monthly",
    "pattern_summary": "string, maximum 25 words, plain-language comparison of this feature across the 6 months",
    "notable_months": ["YYYYMM values that stand out from the account's own pattern; empty if none"],
}

_PROFILE_NOTE_SCHEMA: dict[str, Any] = {
    "field": "occupation | citizenship | indv_org_type",
    "change_summary": "string, maximum 25 words, describe the change only -- never the value as a risk factor",
    "change_dttm": "ISO datetime the change took effect, or null",
    "coincides_with_txn_pattern": "true|false -- does this change's timing overlap a notable_months entry above",
}


def account_assessment_payload(
    monthly_summary: list[dict[str, Any]],
    customer_profile: list[dict[str, Any]],
) -> dict[str, Any]:
    """Ask the model to assess AML risk for one account from its 6-month
    summary and any customer_info history overlapping that window.

    customer_profile entries are expected to already carry resolved labels
    (occupation, citizenship) rather than raw codes -- see
    AnalysisService._customer_profile_for_llm, which is the single place
    that resolution happens.
    """
    task = (
        "Review this account's 6-month monthly summary together with its linked customer profile history as "
        "one case, following the monthly_comparison -> profile_timeline_notes -> findings order and rules "
        "given in the system prompt."
    )
    response_schema: dict[str, Any] = {
        "monthly_comparison": [_MONTHLY_COMPARISON_SCHEMA],
        "profile_timeline_notes": [_PROFILE_NOTE_SCHEMA],
        "findings": [_FINDING_SCHEMA],
        "executive_summary": "string, maximum 120 words",
        "limitations": ["string, maximum 20 words"],
    }
    return {
        "task": task,
        "monthly_summary": monthly_summary,
        "customer_profile": customer_profile,
        "response_schema": response_schema,
    }