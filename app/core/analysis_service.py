from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Literal, TypeVar
from uuid import uuid4

from pydantic import BaseModel, Field, ValidationError

from app.config import Settings
from app.core.llm_work_queue import LlmWorkQueue
from app.core.models import AccountAnalysisRequest, AccountAssessment, AccountFinding, LlmClient, StrList
from app.core.prompts import ANALYST_SYSTEM_PROMPT, account_assessment_payload

logger = logging.getLogger("app.analysis_service")

ModelType = TypeVar("ModelType", bound=BaseModel)


class ModelOutputError(ValueError):
    """Raised when the LLM response is not valid for the required schema."""


class AnalysisService:
    """LLM-led AML workflow over a pre-aggregated monthly account summary.

    Single LLM call per account: no chunking, no cross-chunk synthesis, and
    no separate raw-evidence verification pass -- that machinery existed to
    handle thousands of raw, unstructured transaction rows, which no longer
    applies to a 6-row monthly summary. This service only builds the payload
    and validates that cited year_month values exist in the supplied rows;
    it never scores or computes anomalies itself.
    """

    def __init__(self, llm: LlmClient, settings: Settings, llm_queue: LlmWorkQueue) -> None:
        self._llm = llm
        self._settings = settings
        self._llm_queue = llm_queue

    async def analyze_account(self, request: AccountAnalysisRequest) -> AccountAssessment:
        valid_year_months = {row.year_month for row in request.monthly_summary}
        payload = account_assessment_payload(
            monthly_summary=[row.model_dump(mode="json") for row in request.monthly_summary],
            customer_profile=[record.model_dump(mode="json") for record in request.customer_profile],
        )

        raw_output = await self._ask(request.case_id, "account-assessment", payload, _RawAssessment)

        findings = self._validated_findings(raw_output.findings, valid_year_months)
        # Business rule, not model judgment: low risk closes with human
        # verification; medium/high continue due diligence.
        decision = "close_case" if raw_output.risk_level == "low" else "continue_due_diligence"
        if decision == "continue_due_diligence" and not findings:
            # Never continue due diligence on a risk_level that didn't
            # survive grounding validation for at least one finding.
            decision = "close_case"
            logger.warning(
                "case=%s: decision downgraded to close_case; risk_level=%s but no finding retained a "
                "valid year_month citation",
                request.case_id, raw_output.risk_level,
            )

        return AccountAssessment(
            case_id=request.case_id,
            acct_num=request.acct_num,
            status="needs_review",
            decision=decision,
            risk_level=raw_output.risk_level,
            executive_summary=raw_output.executive_summary,
            findings=findings,
            limitations=list(dict.fromkeys(raw_output.limitations + [
                "Assessment uses only the uploaded monthly summary and linked customer profile history.",
                "LLM output is decision support and requires authorised human review.",
            ])),
            months_reviewed=len(request.monthly_summary),
            generated_at=datetime.now(timezone.utc),
        )

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


class _RawAssessment(BaseModel):
    """Wire shape returned by the LLM, before finding-level grounding validation.

    decision is deliberately not asked of the model: it's a deterministic
    mapping from risk_level (see analyze_account), so the two fields can
    never disagree with each other.
    """
    risk_level: Literal["low", "medium", "high"]
    executive_summary: str = Field(max_length=1_200)
    findings: list[AccountFinding] = Field(default_factory=list, max_length=10)
    limitations: StrList = Field(default_factory=list, max_length=8)