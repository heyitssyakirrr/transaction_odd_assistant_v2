from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field


RiskLevel = Literal["low", "medium", "high"]
FindingSeverity = Literal["medium", "high"]
FindingCategory = Literal[
    "dormancy_reactivation",
    "activity_value_change",
    "debit_credit_flow",
    "burst_and_gaps",
]
ReviewCheckName = Literal[
    "dormancy_reactivation",
    "activity_value_change",
    "debit_credit_flow",
    "burst_and_gaps",
]
ReviewOutcome = Literal["observed", "not_observed", "insufficient_data"]
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


class AccountFinding(BaseModel):
    finding_id: str
    category: FindingCategory
    severity: FindingSeverity
    rationale: str = Field(min_length=1, max_length=360)
    evidence: list[EvidenceItem] = Field(min_length=1, max_length=8)


class ReviewCheck(BaseModel):
    """One mandatory AML review dimension, including a negative result."""

    check: ReviewCheckName
    outcome: ReviewOutcome
    rationale: str = Field(min_length=1, max_length=360)
    evidence: list[EvidenceItem] = Field(default_factory=list, max_length=8)


class ReviewerQuestion(BaseModel):
    question: str = Field(min_length=1, max_length=300)
    evidence: list[EvidenceItem] = Field(default_factory=list, max_length=4)


class AssessmentLimitation(BaseModel):
    limitation: str = Field(min_length=1, max_length=300)
    evidence: list[EvidenceItem] = Field(default_factory=list, max_length=4)


class CustomerProfileContext(BaseModel):
    """Profile-to-activity context and exact supporting source fields."""

    summary: str = Field(min_length=1, max_length=650)
    evidence: list[EvidenceItem] = Field(min_length=1, max_length=16)


class AccountAssessment(BaseModel):
    case_id: str
    acct_num: str
    status: Literal["completed", "needs_review"]
    decision: Literal["close_case", "continue_due_diligence"]
    risk_level: RiskLevel
    executive_summary: str
    review_checks: list[ReviewCheck] = Field(min_length=4, max_length=4)
    findings: list[AccountFinding] = Field(default_factory=list)
    reviewer_questions: list[ReviewerQuestion] = Field(default_factory=list, max_length=3)
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
