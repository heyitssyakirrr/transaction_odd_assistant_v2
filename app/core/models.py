from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field


RiskLevel = Literal["low", "medium", "high"]
FindingCategory = Literal[
    "dormancy_reactivation",
    "activity_spike",
    "flow_imbalance",
    "burst_activity",
    "unusual_variability",
    "profile_timing",
]


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


class MonthlyEvidenceItem(BaseModel):
    """An exact source value, hydrated by the application rather than the model."""

    evidence_id: str
    year_month: str
    feature: str
    value: str


class AccountFinding(BaseModel):
    finding_id: str
    category: FindingCategory
    severity: RiskLevel
    rationale: str = Field(min_length=1, max_length=360)
    evidence: list[MonthlyEvidenceItem] = Field(min_length=1, max_length=4)


class MonthlyComparisonNote(BaseModel):
    title: str = Field(min_length=1, max_length=80)
    pattern_summary: str = Field(min_length=1, max_length=300)
    evidence: list[MonthlyEvidenceItem] = Field(min_length=1, max_length=4)


class ProfileTimelineNote(BaseModel):
    field: Literal["occupation", "citizenship", "indv_org_type"]
    change_summary: str = Field(min_length=1, max_length=220)
    # Keep this as text in the wire contract. Some grammar backends reject
    # JSON Schema's date-time format even though they support normal strings.
    change_dttm: str | None = Field(default=None, max_length=40)
    coincides_with_txn_pattern: bool = False


class AccountAssessment(BaseModel):
    case_id: str
    acct_num: str
    status: Literal["completed", "needs_review"]
    decision: Literal["close_case", "continue_due_diligence"]
    risk_level: RiskLevel
    executive_summary: str
    monthly_comparison: list[MonthlyComparisonNote] = Field(default_factory=list)
    profile_notes: list[ProfileTimelineNote] = Field(default_factory=list)
    findings: list[AccountFinding] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    months_reviewed: int
    profile_records_matched: int
    generated_at: datetime


class LlmClient(Protocol):
    async def complete_json(
        self,
        *,
        system_prompt: str,
        user_payload: dict[str, Any],
        response_schema: dict[str, Any],
        schema_name: str,
    ) -> dict[str, Any]: ...
