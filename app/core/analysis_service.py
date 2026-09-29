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
    AssessmentLimitation,
    CustomerProfileRecord,
    EvidenceItem,
    FindingCategory,
    FindingSeverity,
    LlmClient,
    ReviewCheck,
    ReviewCheckName,
    ReviewOutcome,
    ReviewerQuestion,
    RiskLevel,
)
from app.core.prompts import ANALYST_SYSTEM_PROMPT, account_assessment_payload
from app.core.reference_data import resolve_citizenship, resolve_occupation

logger = logging.getLogger("app.analysis_service")
ModelType = TypeVar("ModelType", bound=BaseModel)
_SEVERITY_RANK: dict[RiskLevel, int] = {"low": 0, "medium": 1, "high": 2}
_REQUIRED_REVIEW_CHECKS: set[ReviewCheckName] = {
    "dormancy_reactivation",
    "activity_value_change",
    "debit_credit_flow",
    "burst_and_gaps",
    "profile_consistency",
}
_FINDING_CHECK: dict[FindingCategory, ReviewCheckName] = {
    "dormancy_reactivation": "dormancy_reactivation",
    "activity_spike": "activity_value_change",
    "flow_imbalance": "debit_credit_flow",
    "burst_activity": "burst_and_gaps",
    "unusual_variability": "activity_value_change",
    "profile_activity_mismatch": "profile_consistency",
}


class ModelOutputError(ValueError):
    """The model did not produce a complete, schema-valid, grounded assessment."""


class AnalysisService:
    """One grounded LLM assessment for one six-month account review.

    The application only serializes source facts, verifies citations, and
    enforces response consistency. It does not calculate AML risk signals or
    make a customer-risk decision on the model's behalf.
    """

    def __init__(self, llm: LlmClient, settings: Settings, llm_queue: LlmWorkQueue) -> None:
        self._llm = llm
        self._settings = settings
        self._llm_queue = llm_queue

    async def analyze_account(self, request: AccountAnalysisRequest) -> AccountAssessment:
        profile_payload = self._profile_records_for_llm(request.customer_profile)
        evidence_catalog = self._evidence_catalog(request, profile_payload)
        payload = account_assessment_payload(
            monthly_summary=[row.model_dump(mode="json") for row in request.monthly_summary],
            customer_profile=profile_payload,
            available_evidence_ids=list(evidence_catalog),
        )
        raw = await self._ask(request.case_id, payload, _RawAssessment)
        assessment = self._hydrate_and_validate(raw, evidence_catalog)

        return AccountAssessment(
            case_id=request.case_id,
            acct_num=request.acct_num,
            status="needs_review",
            decision="close_case" if raw.risk_level == "low" else "continue_due_diligence",
            risk_level=raw.risk_level,
            executive_summary=raw.executive_summary,
            review_checks=assessment["review_checks"],
            findings=assessment["findings"],
            reviewer_questions=assessment["reviewer_questions"],
            limitations=assessment["limitations"] + [
                AssessmentLimitation(limitation="Assessment uses only six monthly aggregates and linked customer-profile history."),
                AssessmentLimitation(limitation="No counterparty, transaction narrative, channel, transaction sequence, expected turnover, or declared income was available."),
                AssessmentLimitation(limitation="Citizenship is not used as a transaction-risk factor without geographical transaction or sanctions data."),
                AssessmentLimitation(limitation="LLM output is decision support and requires authorised human review."),
            ],
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

    def _profile_records_for_llm(
        self, records: list[CustomerProfileRecord]
    ) -> list[dict[str, str | None]]:
        """Assign stable, non-sensitive citation IDs to full profile history."""
        return [
            {
                "profile_record_id": f"P{index}",
                "occupation_cd": record.occupation_cd,
                "occupation": resolve_occupation(record.occupation_cd, self._settings.occupation_code_path),
                "citizen_cd": record.citizen_cd,
                "citizenship": resolve_citizenship(record.citizen_cd),
                "indv_org_type": record.indv_org_type,
                "last_maint_dt": record.last_maint_dt.isoformat() if record.last_maint_dt else None,
                "valid_from_dttm": record.valid_from_dttm.isoformat(),
                "valid_to_dttm": record.valid_to_dttm.isoformat() if record.valid_to_dttm else None,
            }
            for index, record in enumerate(records, start=1)
        ]

    @staticmethod
    def _evidence_catalog(
        request: AccountAnalysisRequest,
        profile_payload: list[dict[str, str | None]],
    ) -> dict[str, EvidenceItem]:
        catalog: dict[str, EvidenceItem] = {}
        excluded = {"acct_num", "year_month"}
        for row in request.monthly_summary:
            for feature in type(row).model_fields:
                if feature in excluded:
                    continue
                evidence_id = f"M{row.year_month}.{feature}"
                catalog[evidence_id] = EvidenceItem(
                    evidence_id=evidence_id,
                    source="monthly_summary",
                    label=f"{row.year_month} · {feature}",
                    value=str(getattr(row, feature)),
                )

        profile_labels = {
            "occupation_cd": "Occupation code",
            "occupation": "Occupation",
            "citizen_cd": "Citizenship code",
            "citizenship": "Citizenship",
            "indv_org_type": "Individual / organisation type",
            "last_maint_dt": "Profile maintenance timestamp",
            "valid_from_dttm": "Profile effective from",
            "valid_to_dttm": "Profile effective to",
        }
        for profile in profile_payload:
            record_id = profile["profile_record_id"]
            for field, label in profile_labels.items():
                value = profile.get(field)
                if value is None:
                    continue
                evidence_id = f"{record_id}.{field}"
                catalog[evidence_id] = EvidenceItem(
                    evidence_id=evidence_id,
                    source="customer_profile",
                    label=f"{record_id} · {label}",
                    value=value,
                )
        return catalog

    @staticmethod
    def _hydrate_and_validate(
        raw: "_RawAssessment", catalog: dict[str, EvidenceItem]
    ) -> dict[str, list[ReviewCheck] | list[AccountFinding] | list[ReviewerQuestion] | list[AssessmentLimitation]]:
        def resolve(ids: list[str], item_label: str) -> list[EvidenceItem]:
            unknown = [evidence_id for evidence_id in ids if evidence_id not in catalog]
            if unknown:
                raise ModelOutputError(
                    f"LLM cited evidence not present in the supplied data for {item_label}: {', '.join(unknown)}"
                )
            return [catalog[evidence_id] for evidence_id in dict.fromkeys(ids)]

        review_checks = [
            ReviewCheck(
                check=check_name,
                outcome=item.outcome,
                rationale=item.rationale,
                evidence=resolve(item.evidence_ids, f"review check '{check_name}'"),
            )
            for check_name, item in raw.review_checks.ordered_items()
        ]
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
        reviewer_questions = [
            ReviewerQuestion(
                question=item.question,
                evidence=resolve(item.evidence_ids, "reviewer question"),
            )
            for item in raw.reviewer_questions
        ]
        limitations = [
            AssessmentLimitation(
                limitation=item.limitation,
                evidence=resolve(item.evidence_ids, "limitation"),
            )
            for item in raw.limitations
        ]
        return {
            "review_checks": review_checks,
            "findings": findings,
            "reviewer_questions": reviewer_questions,
            "limitations": limitations,
        }

    @staticmethod
    def _rollup_risk_level(findings: list[Any]) -> RiskLevel:
        if not findings:
            return "low"
        return max(findings, key=lambda finding: _SEVERITY_RANK[finding.severity]).severity


class _EvidenceReferences(BaseModel):
    model_config = ConfigDict(extra="forbid")

    evidence_ids: list[str] = Field(default_factory=list, max_length=4)


class _RawReviewCheck(_EvidenceReferences):
    outcome: ReviewOutcome
    rationale: str = Field(min_length=1, max_length=360)


class _RawReviewChecks(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dormancy_reactivation: _RawReviewCheck
    activity_value_change: _RawReviewCheck
    debit_credit_flow: _RawReviewCheck
    burst_and_gaps: _RawReviewCheck
    profile_consistency: _RawReviewCheck

    def ordered_items(self) -> list[tuple[ReviewCheckName, _RawReviewCheck]]:
        return [
            ("dormancy_reactivation", self.dormancy_reactivation),
            ("activity_value_change", self.activity_value_change),
            ("debit_credit_flow", self.debit_credit_flow),
            ("burst_and_gaps", self.burst_and_gaps),
            ("profile_consistency", self.profile_consistency),
        ]


class _RawFinding(_EvidenceReferences):
    category: FindingCategory
    severity: FindingSeverity
    rationale: str = Field(min_length=1, max_length=360)


class _RawReviewerQuestion(_EvidenceReferences):
    question: str = Field(min_length=1, max_length=300)


class _RawLimitation(_EvidenceReferences):
    limitation: str = Field(min_length=1, max_length=300)


class _RawAssessment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    risk_level: RiskLevel
    executive_summary: str = Field(min_length=1, max_length=600)
    review_checks: _RawReviewChecks
    findings: list[_RawFinding] = Field(default_factory=list, max_length=3)
    reviewer_questions: list[_RawReviewerQuestion] = Field(default_factory=list, max_length=3)
    limitations: list[_RawLimitation] = Field(default_factory=list, max_length=3)

    @model_validator(mode="after")
    def validate_decision_contract(self) -> "_RawAssessment":
        for check_name, item in self.review_checks.ordered_items():
            if check_name != "profile_consistency" and not item.evidence_ids:
                raise ValueError(f"review check '{check_name}' requires at least one evidence ID")

        observed = {
            check_name for check_name, item in self.review_checks.ordered_items()
            if item.outcome == "observed"
        }
        for finding in self.findings:
            if _FINDING_CHECK[finding.category] not in observed:
                raise ValueError(
                    f"finding '{finding.category}' requires its corresponding review check to be observed"
                )

        expected_risk = AnalysisService._rollup_risk_level(self.findings)
        if self.risk_level != expected_risk:
            raise ValueError(
                "risk_level must equal the highest finding severity, or low when findings is empty"
            )
        if self.risk_level == "high" and len(self.findings) < 2:
            raise ValueError("high risk requires at least two corroborating material findings")
        return self
