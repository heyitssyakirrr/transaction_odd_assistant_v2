from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Literal, TypeVar
from uuid import uuid4

from pydantic import BaseModel, Field, ValidationError

from app.config import Settings
from app.core.llm_work_queue import LlmWorkQueue
from app.core.models import (
    AccountAnalysisRequest,
    AccountAssessment,
    AccountFinding,
    CustomerProfileRecord,
    LlmClient,
    MonthlyComparisonNote,
    ProfileTimelineNote,
    StrList,
)
from app.core.prompts import ANALYST_SYSTEM_PROMPT, account_assessment_payload
from app.core.reference_data import resolve_citizenship, resolve_occupation

logger = logging.getLogger("app.analysis_service")

ModelType = TypeVar("ModelType", bound=BaseModel)

# Ranking used to roll findings up into an overall risk_level. "critical" is
# a valid per-finding severity but AccountAssessment.risk_level only allows
# low/medium/high, so it collapses into "high" at the top of the scale.
_SEVERITY_RANK: dict[str, int] = {"low": 0, "medium": 1, "high": 2, "critical": 3}


class ModelOutputError(ValueError):
    """Raised when the LLM response is not valid for the required schema."""


class AnalysisService:
    """LLM-led AML workflow over a pre-aggregated monthly account summary.

    Single LLM call per account. The LLM does all of the analytical
    reasoning -- comparing months, correlating profile changes, deciding
    findings and their severity. This service does not compute statistics
    or scores itself; it only:
      - resolves customer_profile codes to labels before the call (the LLM
        cannot reason about opaque codes), and
      - applies fixed, auditable business rules AFTER the call: risk_level
        is a deterministic rollup of the findings' own severities (never
        asked of the model as an independent field, so it can't disagree
        with the findings that are supposed to justify it), and decision
        is a deterministic mapping from risk_level.
    """

    def __init__(self, llm: LlmClient, settings: Settings, llm_queue: LlmWorkQueue) -> None:
        self._llm = llm
        self._settings = settings
        self._llm_queue = llm_queue

    async def analyze_account(self, request: AccountAnalysisRequest) -> AccountAssessment:
        valid_year_months = {row.year_month for row in request.monthly_summary}
        payload = account_assessment_payload(
            monthly_summary=[row.model_dump(mode="json") for row in request.monthly_summary],
            customer_profile=[self._customer_profile_for_llm(record) for record in request.customer_profile],
        )

        raw_output = await self._ask(request.case_id, "account-assessment", payload, _RawAssessment)

        findings = self._validated_findings(raw_output.findings, valid_year_months)
        monthly_comparison = self._validated_comparison(raw_output.monthly_comparison, valid_year_months)

        # Business rule, not model judgment: risk_level is derived from the
        # findings' own severities, so it can never contradict them the way
        # an independently-asked risk_level field could.
        risk_level = self._rollup_risk_level(findings)
        decision = "close_case" if risk_level == "low" else "continue_due_diligence"

        return AccountAssessment(
            case_id=request.case_id,
            acct_num=request.acct_num,
            status="needs_review",
            decision=decision,
            risk_level=risk_level,
            executive_summary=raw_output.executive_summary,
            monthly_comparison=monthly_comparison,
            profile_notes=raw_output.profile_timeline_notes,
            findings=findings,
            limitations=list(dict.fromkeys(raw_output.limitations + [
                "Assessment uses only the uploaded monthly summary and linked customer profile history.",
                "LLM output is decision support and requires authorised human review.",
            ])),
            months_reviewed=len(request.monthly_summary),
            generated_at=datetime.now(timezone.utc),
        )

    def _customer_profile_for_llm(self, record: CustomerProfileRecord) -> dict[str, Any]:
        """Resolve coded fields to labels the LLM can reason about.

        Deliberately drops acct_num/customer_num (redundant, and unneeded
        for the analytical task) and the raw occupation_cd/citizen_cd
        (opaque to the model). Only the lookup RESULT is sent, never the
        lookup table itself, so payload size doesn't grow with the size of
        the occupation CSV.
        """
        return {
            "occupation": resolve_occupation(record.occupation_cd, self._settings.occupation_code_path),
            "citizenship": resolve_citizenship(record.citizen_cd),
            "indv_org_type": record.indv_org_type,
            "last_maint_dt": record.last_maint_dt.isoformat() if record.last_maint_dt else None,
            "valid_from_dttm": record.valid_from_dttm.isoformat(),
            "valid_to_dttm": record.valid_to_dttm.isoformat() if record.valid_to_dttm else None,
        }

    async def _ask(self, case_id: str, stage: str, payload: dict[str, object], model: type[ModelType]) -> ModelType:
        raw = await self._llm_queue.submit(
            name=f"case={case_id} stage={stage}",
            operation=lambda: self._llm.complete_json(system_prompt=ANALYST_SYSTEM_PROMPT, user_payload=payload),
        )
        try:
            if set(raw) == {"response_schema"} and isinstance(raw["response_schema"], dict):
                raw = raw["response_schema"]
            return model.model_validate(raw)
        except ValidationError as exc:
            raise ModelOutputError(f"LLM JSON does not match the required schema: {exc}") from exc

    @staticmethod
    def _validated_findings(findings: list[AccountFinding], allowed_year_months: set[str]) -> list[AccountFinding]:
        retained: list[AccountFinding] = []
        for finding in findings:
            evidence = [item for item in finding.evidence if item.year_month in allowed_year_months]
            if evidence:
                finding.finding_id = finding.finding_id or str(uuid4())
                finding.evidence = evidence
                retained.append(finding)
        return retained

    @staticmethod
    def _validated_comparison(
        notes: list[MonthlyComparisonNote], allowed_year_months: set[str]
    ) -> list[MonthlyComparisonNote]:
        """Drop any notable_months the model cited that weren't actually
        supplied -- same grounding discipline as findings evidence."""
        retained: list[MonthlyComparisonNote] = []
        for note in notes:
            note.notable_months = [ym for ym in note.notable_months if ym in allowed_year_months]
            retained.append(note)
        return retained

    @staticmethod
    def _rollup_risk_level(findings: list[AccountFinding]) -> Literal["low", "medium", "high"]:
        if not findings:
            return "low"
        worst = max(_SEVERITY_RANK[finding.severity] for finding in findings)
        if worst >= _SEVERITY_RANK["high"]:
            return "high"
        if worst >= _SEVERITY_RANK["medium"]:
            return "medium"
        return "low"


class _RawAssessment(BaseModel):
    """Wire shape returned by the LLM, before grounding validation.

    risk_level is deliberately NOT part of this shape: it's a deterministic
    rollup of findings[].severity (see AnalysisService._rollup_risk_level),
    computed after grounding validation runs, so it can never be asked of
    (and disagree with) the model independently. decision is likewise never
    asked of the model -- it's a deterministic mapping from risk_level.
    """
    monthly_comparison: list[MonthlyComparisonNote] = Field(default_factory=list, max_length=8)
    profile_timeline_notes: list[ProfileTimelineNote] = Field(default_factory=list, max_length=8)
    findings: list[AccountFinding] = Field(default_factory=list, max_length=10)
    executive_summary: str = Field(max_length=1_200)
    limitations: StrList = Field(default_factory=list, max_length=8)
