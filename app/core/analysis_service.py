from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, TypeVar
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from app.adapters.llm_client import LlmOutputFormatError
from app.config import Settings
from app.core.llm_work_queue import LlmWorkQueue
from app.core.models import (
    AccountAnalysisRequest, AccountAssessment, AccountFinding, AssessmentLimitation,
    CustomerProfileRecord, EvidenceItem, FindingCategory, FindingSeverity, LlmClient,
    ReviewCheck, ReviewCheckName, ReviewOutcome, ReviewerQuestion, RiskLevel,
)
from app.core.prompts import ANALYST_SYSTEM_PROMPT, FORMAT_RETRY_SYSTEM_PROMPT, account_assessment_input
from app.core.reference_data import resolve_citizenship, resolve_occupation

logger = logging.getLogger("app.analysis_service")
ModelType = TypeVar("ModelType", bound=BaseModel)
_CHECK_ORDER: tuple[ReviewCheckName, ...] = (
    "dormancy_reactivation", "activity_value_change", "debit_credit_flow",
    "burst_and_gaps", "profile_consistency",
)
_CHECK_CATEGORY: dict[ReviewCheckName, FindingCategory] = {name: name for name in _CHECK_ORDER}
_SEVERITY_RANK: dict[RiskLevel, int] = {"low": 0, "medium": 1, "high": 2}


class ModelOutputError(ValueError):
    """A model response was not a complete, grounded assessment."""


class AnalysisService:
    """One complete LLM assessment with strict validation and a format-only retry.

    The application serialises evidence and validates citations. It never scores
    transaction data or invents an AML finding on the model's behalf.
    """

    def __init__(self, llm: LlmClient, settings: Settings, llm_queue: LlmWorkQueue) -> None:
        self._llm = llm
        self._settings = settings
        self._llm_queue = llm_queue

    async def analyze_account(self, request: AccountAnalysisRequest) -> AccountAssessment:
        profile = self._profile_records_for_llm(request.customer_profile)
        catalog = self._evidence_catalog(request, profile)
        prompt_input = account_assessment_input(
            [row.model_dump(mode="json") for row in request.monthly_summary], profile, list(catalog),
        )
        try:
            return await self._analyze_with_format_retry(request, catalog, prompt_input)
        except (LlmOutputFormatError, ModelOutputError) as exc:
            logger.error("All model-output attempts failed: case=%s error=%s", request.case_id, exc)
            return self._manual_review_fallback(request, str(exc))

    async def _analyze_with_format_retry(
        self, request: AccountAnalysisRequest, catalog: dict[str, EvidenceItem], prompt_input: str,
    ) -> AccountAssessment:
        try:
            raw = await self._ask(request.case_id, "account-assessment", ANALYST_SYSTEM_PROMPT, prompt_input)
            return self._hydrate_assessment(request, raw, catalog)
        except (LlmOutputFormatError, ModelOutputError) as first_error:
            logger.warning("Assessment output rejected; retrying once with format gate: case=%s error=%s", request.case_id, first_error)
            try:
                raw = await self._ask(request.case_id, "account-assessment-format-retry", FORMAT_RETRY_SYSTEM_PROMPT, prompt_input)
                return self._hydrate_assessment(request, raw, catalog)
            except (LlmOutputFormatError, ModelOutputError) as second_error:
                raise ModelOutputError("Qwen did not produce a verifiable AML assessment after one format retry.") from second_error

    async def _ask(self, case_id: str, stage: str, system_prompt: str, prompt_input: str) -> "_RawAssessment":
        result = await self._llm_queue.submit(
            name=f"case={case_id} stage={stage}",
            operation=lambda: self._llm.complete_json(
                system_prompt=system_prompt, user_payload=prompt_input,
                response_schema=_RawAssessment.model_json_schema(), schema_name="account_assessment",
                max_response_tokens=self._settings.max_response_tokens,
            ),
        )
        try:
            return _RawAssessment.model_validate(result)
        except ValidationError as exc:
            raise ModelOutputError(f"LLM output did not meet the assessment schema: {exc}") from exc

    def _hydrate_assessment(self, request: AccountAnalysisRequest, raw: "_RawAssessment", catalog: dict[str, EvidenceItem]) -> AccountAssessment:
        checks = [
            ReviewCheck(check=name, outcome=item.outcome, rationale=item.rationale,
                        evidence=self._resolve(item.evidence_ids, catalog, f"review check '{name}'"))
            for name, item in raw.review_checks.ordered_items()
        ]
        observed = {check.check for check in checks if check.outcome == "observed"}
        findings = [
            AccountFinding(
                finding_id=str(uuid4()), category=item.category, severity=item.severity,
                rationale=item.rationale, evidence=self._resolve(item.evidence_ids, catalog, f"finding '{item.category}'"),
            ) for item in raw.findings
        ]
        for finding in findings:
            if finding.category not in observed:
                raise ModelOutputError(f"finding '{finding.category}' has no observed supporting review check")
        if "profile_consistency" in {finding.category for finding in findings}:
            if not any(finding.category != "profile_consistency" for finding in findings):
                raise ModelOutputError("profile consistency cannot be a standalone material finding")
        expected_risk: RiskLevel = max((item.severity for item in raw.findings), key=lambda value: _SEVERITY_RANK[value], default="low")
        if raw.risk_level != expected_risk:
            raise ModelOutputError("risk_level must equal the highest finding severity, or low with no findings")
        if raw.risk_level == "high" and len(findings) < 2:
            raise ModelOutputError("high risk requires two corroborating findings")
        questions = [ReviewerQuestion(question=item.question, evidence=self._resolve(item.evidence_ids, catalog, "reviewer question")) for item in raw.reviewer_questions]
        return AccountAssessment(
            case_id=request.case_id, acct_num=request.acct_num, status="completed",
            decision="close_case" if raw.risk_level == "low" else "continue_due_diligence",
            risk_level=raw.risk_level, executive_summary=raw.executive_summary,
            review_checks=checks, findings=findings, reviewer_questions=questions,
            limitations=self._static_limitations(), months_reviewed=len(request.monthly_summary),
            profile_records_matched=len(request.customer_profile), generated_at=datetime.now(timezone.utc),
        )

    def _manual_review_fallback(self, request: AccountAnalysisRequest, error: str) -> AccountAssessment:
        """Never convert an unverifiable model answer into a low-risk closure."""
        checks = [ReviewCheck(check=name, outcome="insufficient_data", rationale="Model assessment output could not be verified; no conclusion is shown.") for name in _CHECK_ORDER]
        return AccountAssessment(
            case_id=request.case_id, acct_num=request.acct_num, status="needs_review",
            decision="continue_due_diligence", risk_level="medium",
            executive_summary="The LLM did not produce a verifiable assessment after one format retry. Do not close this case on the basis of this result.",
            review_checks=checks, findings=[], reviewer_questions=[],
            limitations=self._static_limitations() + [AssessmentLimitation(limitation="Model output verification failed; an authorised reviewer must assess the supplied account data directly.")],
            months_reviewed=len(request.monthly_summary), profile_records_matched=len(request.customer_profile),
            generated_at=datetime.now(timezone.utc),
        )

    @staticmethod
    def _static_limitations() -> list[AssessmentLimitation]:
        return [
            AssessmentLimitation(limitation="Assessment uses only six monthly aggregates and linked customer-profile history."),
            AssessmentLimitation(limitation="No counterparty, transaction narrative, channel, transaction sequence, expected turnover, declared income, or source-of-funds data was available."),
            AssessmentLimitation(limitation="Citizenship is not used as a transaction-risk factor without geographical transaction or sanctions data."),
            AssessmentLimitation(limitation="LLM output is decision support and requires authorised human review."),
        ]

    def _profile_records_for_llm(self, records: list[CustomerProfileRecord]) -> list[dict[str, str | None]]:
        return [{
            "profile_record_id": f"P{index}", "occupation_cd": record.occupation_cd,
            "occupation": resolve_occupation(record.occupation_cd, self._settings.occupation_code_path),
            "citizen_cd": record.citizen_cd, "citizenship": resolve_citizenship(record.citizen_cd),
            "indv_org_type": record.indv_org_type,
            "last_maint_dt": record.last_maint_dt.isoformat() if record.last_maint_dt else None,
            "valid_from_dttm": record.valid_from_dttm.isoformat(),
            "valid_to_dttm": record.valid_to_dttm.isoformat() if record.valid_to_dttm else None,
        } for index, record in enumerate(records, start=1)]

    @staticmethod
    def _evidence_catalog(request: AccountAnalysisRequest, profile: list[dict[str, str | None]]) -> dict[str, EvidenceItem]:
        catalog: dict[str, EvidenceItem] = {}
        for row in request.monthly_summary:
            for field in type(row).model_fields:
                if field not in {"acct_num", "year_month"}:
                    item_id = f"M{row.year_month}.{field}"
                    catalog[item_id] = EvidenceItem(evidence_id=item_id, source="monthly_summary", label=f"{row.year_month} · {field}", value=str(getattr(row, field)))
        labels = {
            "occupation_cd": "Occupation code", "occupation": "Occupation", "citizen_cd": "Citizenship code", "citizenship": "Citizenship",
            "indv_org_type": "Individual / organisation type", "last_maint_dt": "Profile maintenance timestamp",
            "valid_from_dttm": "Profile effective from", "valid_to_dttm": "Profile effective to",
        }
        for record in profile:
            for field, label in labels.items():
                if (value := record.get(field)) is not None:
                    item_id = f"{record['profile_record_id']}.{field}"
                    catalog[item_id] = EvidenceItem(evidence_id=item_id, source="customer_profile", label=f"{record['profile_record_id']} · {label}", value=value)
        return catalog

    @staticmethod
    def _resolve(ids: list[str], catalog: dict[str, EvidenceItem], label: str) -> list[EvidenceItem]:
        unknown = [item_id for item_id in ids if item_id not in catalog]
        if unknown:
            raise ModelOutputError(f"LLM cited evidence not present in supplied data for {label}: {', '.join(unknown)}")
        return [catalog[item_id] for item_id in dict.fromkeys(ids)]


class _EvidenceReferences(BaseModel):
    model_config = ConfigDict(extra="forbid")
    evidence_ids: list[str] = Field(default_factory=list, max_length=4)


class _RawReviewCheck(_EvidenceReferences):
    outcome: ReviewOutcome
    rationale: str = Field(min_length=1, max_length=180)


class _RawReviewChecks(BaseModel):
    model_config = ConfigDict(extra="forbid")
    dormancy_reactivation: _RawReviewCheck
    activity_value_change: _RawReviewCheck
    debit_credit_flow: _RawReviewCheck
    burst_and_gaps: _RawReviewCheck
    profile_consistency: _RawReviewCheck

    def ordered_items(self) -> list[tuple[ReviewCheckName, _RawReviewCheck]]:
        return [(name, getattr(self, name)) for name in _CHECK_ORDER]


class _RawFinding(_EvidenceReferences):
    category: FindingCategory
    severity: FindingSeverity
    rationale: str = Field(min_length=1, max_length=180)


class _RawReviewerQuestion(_EvidenceReferences):
    question: str = Field(min_length=1, max_length=240)


class _RawAssessment(BaseModel):
    model_config = ConfigDict(extra="forbid")
    risk_level: RiskLevel
    executive_summary: str = Field(min_length=1, max_length=380)
    review_checks: _RawReviewChecks
    findings: list[_RawFinding] = Field(default_factory=list, max_length=2)
    reviewer_questions: list[_RawReviewerQuestion] = Field(default_factory=list, max_length=2)

    @model_validator(mode="after")
    def validate_schema_contract(self) -> "_RawAssessment":
        for name, item in self.review_checks.ordered_items():
            if name != "profile_consistency" and not item.evidence_ids:
                raise ValueError(f"review check '{name}' requires evidence")
            if name == "profile_consistency" and item.outcome == "observed":
                if not any(value.startswith("M") for value in item.evidence_ids) or not any(value.startswith("P") for value in item.evidence_ids):
                    raise ValueError("observed profile consistency requires monthly and profile evidence")
        return self
