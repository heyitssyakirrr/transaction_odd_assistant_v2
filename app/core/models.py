from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field


RiskLevel = Literal["low", "medium", "high"]
ReviewCheckName = Literal[
    "activity_after_inactivity",
    "activity_and_amount_change",
    "money_in_and_out",
    "burst_and_gaps",
]
ReviewOutcome = Literal["pattern_found", "no_pattern_found", "not_verified"]
Decision = Literal["further_review_suggested", "no_further_review_suggested"]
ProfileEvidenceSource = Literal["monthly_summary", "customer_profile"]


class MonthlySummaryRow(BaseModel):
    """One upstream-provided aggregate row; this service does not score it."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    acct_num: str = Field(min_length=1, max_length=64)
    year_month: str = Field(pattern=r"^\d{6}$")
    txn_count_monthly: int = Field(ge=0)
    pct_burst: float
    total_amount: Decimal
    avg_amount: Decimal
    std_amount: Decimal
    max_amount: Decimal
    pct_trx_gap: float
    monthly_debit: Decimal
    monthly_credit: Decimal
    debit_count_monthly: int = Field(ge=0)
    credit_count_monthly: int = Field(ge=0)
    monthly_avg_debit: Decimal
    monthly_avg_credit: Decimal


class CustomerProfileRecord(BaseModel):
    """One SCD2 customer profile version, retained for the account's full history."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    acct_num: str
    customer_num: str
    occupation_cd: str | None = None
    citizen_cd: str | None = None
    indv_org_type: str | None = None
    last_maint_dt: datetime | None = None
    valid_from_dttm: datetime
    valid_to_dttm: datetime | None = None


class AccountAnalysisRequest(BaseModel):
    case_id: str = Field(min_length=1, max_length=128)
    source_filename: str = Field(min_length=1, max_length=255)
    acct_num: str = Field(min_length=1, max_length=64)
    monthly_summary: list[MonthlySummaryRow] = Field(min_length=6, max_length=6)
    customer_profile: list[CustomerProfileRecord] = Field(default_factory=list)


class EvidenceItem(BaseModel):
    """An exact source value hydrated by the application, never invented by the model."""

    evidence_id: str
    source: ProfileEvidenceSource
    label: str
    value: str


class EvidenceTable(BaseModel):
    """Exact CSV/profile values laid out for staff. Built by the application, never by the model."""

    columns: list[str] = Field(min_length=1)
    rows: list[list[str]] = Field(default_factory=list)
    # Row indexes the model selected for this check, shown in bold.
    highlighted_rows: list[int] = Field(default_factory=list)


class ReviewCheck(BaseModel):
    """One mandatory review check, including a negative result."""

    check: ReviewCheckName
    title: str
    outcome: ReviewOutcome
    # The model's plain-language label for the kind of change, e.g. "Amounts changed".
    pattern: str | None = None
    months: list[str] = Field(default_factory=list)
    # The model's explanation for staff; None when it was not provided or did not match the figures.
    insight: str | None = Field(default=None, max_length=400)
    # shown, not_provided (the model gave no sentence) or hidden (it did not match the figures).
    insight_status: Literal["shown", "not_provided", "hidden"] = "not_provided"
    # A short factual line built from the CSV rows.
    facts: str = Field(min_length=1, max_length=400)
    table: EvidenceTable | None = None
    evidence: list[EvidenceItem] = Field(default_factory=list, max_length=8)


class AssessmentLimitation(BaseModel):
    limitation: str = Field(min_length=1, max_length=300)
    evidence: list[EvidenceItem] = Field(default_factory=list, max_length=4)


class CustomerProfileContext(BaseModel):
    """Profile-to-activity context and exact supporting source fields."""

    title: str = "Customer profile vs activity"
    summary: str = Field(min_length=1, max_length=650)
    table: EvidenceTable | None = None
    evidence: list[EvidenceItem] = Field(min_length=1, max_length=16)


class OverallSummary(BaseModel):
    """What staff read first. Written by the summary LLM call from the checked results."""

    # False when the summary call failed and the application could not supply an LLM view.
    verified: bool
    headline: str = Field(min_length=1, max_length=200)
    points: list[str] = Field(default_factory=list, max_length=3)
    why_it_matters: str | None = None
    verify_first: list[str] = Field(default_factory=list, max_length=2)
    risk_reason: str | None = None


class AccountAssessment(BaseModel):
    case_id: str
    acct_num: str
    status: Literal["completed", "needs_review"]
    decision: Decision
    risk_level: RiskLevel
    overall: OverallSummary
    review_checks: list[ReviewCheck] = Field(min_length=4, max_length=4)
    customer_profile_context: CustomerProfileContext | None = None
    limitations: list[AssessmentLimitation] = Field(default_factory=list)
    months_reviewed: int
    profile_records_matched: int
    generated_at: datetime


class LlmClient(Protocol):
    async def complete_json(
        self,
        *,
        system_prompt: str,
        user_payload: dict[str, Any] | str,
        response_schema: dict[str, Any],
        schema_name: str,
        max_response_tokens: int | None = None,
    ) -> dict[str, Any]: ...