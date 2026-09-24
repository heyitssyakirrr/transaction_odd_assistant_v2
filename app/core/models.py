from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Annotated, Any, Literal, Protocol

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field


def _as_list(value: Any) -> Any:
    return [value] if isinstance(value, str) else value


# Small models sometimes return a single string where the schema asks for a list.
StrList = Annotated[list[str], BeforeValidator(_as_list)]


class MonthlySummaryRow(BaseModel):
    """One month of pre-aggregated per-account transaction features.

    Produced upstream by the aggregation job. This service treats every
    numeric field here as-is -- it never derives, recomputes, or scores
    from raw transactions itself.
    """
    model_config = ConfigDict(frozen=True, extra="forbid")

    acct_num: str = Field(min_length=1, max_length=64)
    year_month: str = Field(min_length=6, max_length=6, pattern=r"^\d{6}$")  # "YYYYMM", e.g. 202604    
    txn_count_monthly: int = Field(ge=0)
    pct_burst: float
    total_amount: Decimal
    avg_amount: Decimal
    std_amount: Decimal
    max_amount: Decimal
    #day_gaps: float
    pct_trx_gap: float
    monthly_debit: Decimal
    monthly_credit: Decimal
    debit_count_monthly: int = Field(ge=0)
    credit_count_monthly: int = Field(ge=0)
    monthly_avg_debit: Decimal
    monthly_avg_credit: Decimal


class CustomerProfileRecord(BaseModel):
    """One SCD Type 2 version of a customer's profile from custmer_info.

    Multiple rows per account are expected: each is the profile as it stood
    for [valid_from_dttm, valid_to_dttm). A null valid_to_dttm means current.
    """
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
    monthly_summary: list[MonthlySummaryRow] = Field(min_length=1, max_length=6)
    customer_profile: list[CustomerProfileRecord] = Field(default_factory=list, max_length=50)


class MonthlyEvidenceItem(BaseModel):
    year_month: str
    feature: str
    value: str
    statement: str


class AccountFinding(BaseModel):
    finding_id: str = ""
    category: str
    severity: Literal["low", "medium", "high", "critical"]
    evidence: list[MonthlyEvidenceItem] = Field(default_factory=list)
    rationale: str


class AccountAssessment(BaseModel):
    case_id: str
    acct_num: str
    status: Literal["completed", "needs_review"]
    decision: Literal["close_case", "continue_due_diligence"]
    risk_level: Literal["low", "medium", "high"]
    executive_summary: str
    findings: list[AccountFinding]
    limitations: list[str] = Field(default_factory=list)
    months_reviewed: int
    generated_at: datetime


class LlmClient(Protocol):
    async def complete_json(self, *, system_prompt: str, user_payload: dict[str, Any]) -> dict[str, Any]: ...