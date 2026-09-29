from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, TypeVar
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from app.config import Settings
from app.core.llm_work_queue import LlmWorkQueue
from app.core.models import (
    AccountAnalysisRequest,
    AccountAssessment,
    AccountFinding,
    CustomerProfileRecord,
    FindingCategory,
    LlmClient,
    MonthlyEvidenceItem,
    RiskLevel,
)
from app.core.prompts import ANALYST_SYSTEM_PROMPT, account_assessment_payload
from app.core.reference_data import resolve_citizenship, resolve_occupation

logger = logging.getLogger("app.analysis_service")
ModelType = TypeVar("ModelType", bound=BaseModel)
_SEVERITY_RANK: dict[RiskLevel, int] = {"low": 0, "medium": 1, "high": 2}


class ModelOutputError(ValueError):
    """The model did not produce a complete, schema-valid, grounded assessment."""


class AnalysisService:
    """One strict, evidence-grounded LLM call for a six-month account review.

    This class intentionally does not calculate risk signals. Its deterministic work is
    limited to source-data serialization, evidence-reference verification, and deriving
    a single account risk from the severities the model assigned to its findings.
    """

    def __init__(self, llm: LlmClient, settings: Settings, llm_queue: LlmWorkQueue) -> None:
        self._llm = llm
        self._settings = settings
        self._llm_queue = llm_queue

    async def analyze_account(self, request: AccountAnalysisRequest) -> AccountAssessment:
        evidence_catalog = self._evidence_catalog(request)
        payload = account_assessment_payload(
            monthly_summary=[row.model_dump(mode="json") for row in request.monthly_summary],
            customer_profile=[self._customer_profile_for_llm(record) for record in request.customer_profile],
        )
        raw = await self._ask(request.case_id, payload, _RawAssessment)
        assessment = self._hydrate_and_validate(raw, evidence_catalog)
        risk_level = raw.risk_level

        return AccountAssessment(
            case_id=request.case_id,
            acct_num=request.acct_num,
            status="needs_review",
            decision="close_case" if risk_level == "low" else "continue_due_diligence",
            risk_level=risk_level,
            executive_summary=raw.executive_summary,
            findings=assessment["findings"],
            limitations=list(dict.fromkeys(raw.limitations + [
                "Assessment uses only six monthly aggregates and linked customer-profile history.",
                "No counterparty, transaction narrative, channel, or transaction-level sequence was available.",
                "LLM output is decision support and requires authorised human review.",
            ])),
            months_reviewed=len(request.monthly_summary),
            profile_records_matched=len(request.customer_profile),
            generated_at=datetime.now(timezone.utc),
        )

    async def _ask(self, case_id: str, payload: dict[str, object], model: type[ModelType]) -> ModelType:
        raw = await self._llm_queue.submit(
            name=f"case={case_id} stage=account-assessment",
            operation=lambda: self._llm.complete_json(
                system_prompt=ANALYST_SYSTEM_PROMPT,
                user_payload=payload,
                response_schema=model.model_json_schema(),
                schema_name=model.__name__,
            ),
        )
        try:
            return model.model_validate(raw)
        except ValidationError as exc:
            raise ModelOutputError(f"LLM output did not meet the required schema: {exc}") from exc

    def _customer_profile_for_llm(self, record: CustomerProfileRecord) -> dict[str, Any]:
        return {
            "occupation": resolve_occupation(record.occupation_cd, self._settings.occupation_code_path),
            "citizenship": resolve_citizenship(record.citizen_cd),
            "indv_org_type": record.indv_org_type,
            "last_maint_dt": record.last_maint_dt.isoformat() if record.last_maint_dt else None,
            "valid_from_dttm": record.valid_from_dttm.isoformat(),
            "valid_to_dttm": record.valid_to_dttm.isoformat() if record.valid_to_dttm else None,
        }

    @staticmethod
    def _evidence_catalog(request: AccountAnalysisRequest) -> dict[str, MonthlyEvidenceItem]:
        catalog: dict[str, MonthlyEvidenceItem] = {}
        excluded = {"acct_num", "year_month"}
        for row in request.monthly_summary:
            for feature in type(row).model_fields:
                if feature in excluded:
                    continue
                evidence_id = f"M{row.year_month}.{feature}"
                catalog[evidence_id] = MonthlyEvidenceItem(
                    evidence_id=evidence_id,
                    year_month=row.year_month,
                    feature=feature,
                    value=str(getattr(row, feature)),
                )
        return catalog

    @staticmethod
    def _hydrate_and_validate(
        raw: "_RawAssessment", catalog: dict[str, MonthlyEvidenceItem]
    ) -> dict[str, list[AccountFinding]]:
        def resolve(ids: list[str], item_label: str) -> list[MonthlyEvidenceItem]:
            unknown = [evidence_id for evidence_id in ids if evidence_id not in catalog]
            if unknown:
                raise ModelOutputError(
                    f"LLM cited evidence not present in the uploaded summary for {item_label}: {', '.join(unknown)}"
                )
            return [catalog[evidence_id] for evidence_id in dict.fromkeys(ids)]

        findings = [
            AccountFinding(
                finding_id=str(uuid4()),
                category=item.category,
                severity=item.severity,
                rationale=item.rationale,
                evidence=resolve(item.evidence_ids, f"finding '{item.category}'"),
            )
            for item in raw.findings
        ]
        return {"findings": findings}

    @staticmethod
    def _rollup_risk_level(findings: list[AccountFinding]) -> RiskLevel:
        if not findings:
            return "low"
        return max(findings, key=lambda finding: _SEVERITY_RANK[finding.severity]).severity


class _EvidenceReferences(BaseModel):
    model_config = ConfigDict(extra="forbid")

    evidence_ids: list[str] = Field(min_length=1, max_length=4)


class _RawFinding(_EvidenceReferences):
    category: FindingCategory
    severity: RiskLevel
    rationale: str = Field(min_length=1, max_length=360)


class _RawAssessment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    risk_level: RiskLevel
    executive_summary: str = Field(min_length=1, max_length=600)
    findings: list[_RawFinding] = Field(default_factory=list, max_length=3)
    limitations: list[str] = Field(default_factory=list, max_length=3)

    @model_validator(mode="after")
    def risk_must_match_findings(self) -> "_RawAssessment":
        """Reject internally inconsistent model decisions instead of guessing.

        This is not a risk-scoring rule: the LLM still decides each finding's
        severity.  It only ensures that a response cannot call the account
        low while simultaneously returning a medium/high finding (or vice
        versa), which would make the staff-facing result ambiguous.
        """
        expected = AnalysisService._rollup_risk_level(self.findings)
        if self.risk_level != expected:
            raise ValueError(
                "risk_level must equal the highest severity in findings, or low when findings is empty"
            )
        return self
