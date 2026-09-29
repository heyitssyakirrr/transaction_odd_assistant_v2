from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any, TypeVar
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from app.config import Settings
from app.core.llm_work_queue import LlmWorkQueue
from app.core.models import (
    AccountAnalysisRequest, AccountAssessment, AccountFinding, AssessmentLimitation,
    CustomerProfileRecord, EvidenceItem, FindingCategory, FindingSeverity, LlmClient,
    ReviewCheck, ReviewCheckName, ReviewOutcome, ReviewerQuestion, RiskLevel,
)
from app.core.prompts import (
    FLOW_SYSTEM_PROMPT, PROFILE_SYSTEM_PROMPT, SYNTHESIS_SYSTEM_PROMPT,
    TIMELINE_SYSTEM_PROMPT, stage_payload, synthesis_payload,
)
from app.core.reference_data import resolve_citizenship, resolve_occupation

logger = logging.getLogger("app.analysis_service")
ModelType = TypeVar("ModelType", bound=BaseModel)

_SIGNAL_ORDER: tuple[ReviewCheckName, ...] = (
    "dormancy_reactivation", "activity_value_change", "debit_credit_flow",
    "burst_and_gaps", "profile_consistency",
)
_SIGNAL_IDS = dict(zip(_SIGNAL_ORDER, ("S1", "S2", "S3", "S4", "S5"), strict=True))
_SIGNAL_CATEGORY: dict[ReviewCheckName, FindingCategory] = {
    "dormancy_reactivation": "dormancy_reactivation",
    "activity_value_change": "activity_value_change",
    "debit_credit_flow": "debit_credit_flow",
    "burst_and_gaps": "burst_and_gaps",
    "profile_consistency": "profile_consistency",
}
_TRANSACTION_SIGNALS = set(_SIGNAL_ORDER[:-1])
_SEVERITY_RANK: dict[RiskLevel, int] = {"low": 0, "medium": 1, "high": 2}


class ModelOutputError(ValueError):
    """The model did not produce a complete, schema-valid, grounded assessment."""


class AnalysisService:
    """Four-stage, grounded LLM analysis for one six-month account review.

    The application does not score AML risk. It only serializes source facts,
    enforces citations and decision policy, and renders model-selected signals.
    """

    def __init__(self, llm: LlmClient, settings: Settings, llm_queue: LlmWorkQueue) -> None:
        self._llm = llm
        self._settings = settings
        self._llm_queue = llm_queue

    async def analyze_account(self, request: AccountAnalysisRequest) -> AccountAssessment:
        profile_payload = self._profile_records_for_llm(request.customer_profile)
        catalog = self._evidence_catalog(request, profile_payload)
        monthly = [row.model_dump(mode="json") for row in request.monthly_summary]

        # The independent specialist calls occupy up to the three configured
        # loader slots. Synthesis starts only after all outputs are validated.
        timeline_ids = self._monthly_ids(catalog, self._timeline_fields())
        flow_ids = self._monthly_ids(catalog, self._flow_fields())
        profile_ids = self._profile_stage_ids(catalog)
        timeline, flow, profile = await asyncio.gather(
            self._ask_stage(
                request.case_id, "timeline", TIMELINE_SYSTEM_PROMPT,
                stage_payload(
                    task="Assess dormancy/re-activation and activity/value change.",
                    monthly_summary=monthly, customer_profile=[],
                    available_evidence_ids=timeline_ids,
                ), _RawTimelineStage, self._settings.llm_timeline_max_response_tokens,
            ),
            self._ask_stage(
                request.case_id, "flow", FLOW_SYSTEM_PROMPT,
                stage_payload(
                    task="Assess debit/credit flow and burst/gap behaviour.",
                    monthly_summary=monthly, customer_profile=[],
                    available_evidence_ids=flow_ids,
                ), _RawFlowStage, self._settings.llm_flow_max_response_tokens,
            ),
            self._ask_stage(
                request.case_id, "profile", PROFILE_SYSTEM_PROMPT,
                stage_payload(
                    task="Assess profile timing and potential profile/activity context.",
                    monthly_summary=monthly, customer_profile=profile_payload,
                    available_evidence_ids=profile_ids,
                ), _RawProfileStage, self._settings.llm_profile_max_response_tokens,
            ),
        )
        self._validate_stage_evidence("timeline", (timeline.dormancy_reactivation, timeline.activity_value_change), timeline_ids)
        self._validate_stage_evidence("flow", (flow.debit_credit_flow, flow.burst_and_gaps), flow_ids)
        self._validate_stage_evidence("profile", (profile.profile_consistency,), profile_ids)
        raw_checks = self._ordered_raw_checks(timeline, flow, profile)
        checks = self._hydrate_checks(raw_checks, catalog)

        synthesis = await self._ask_stage(
            request.case_id, "synthesis", SYNTHESIS_SYSTEM_PROMPT,
            synthesis_payload(self._signal_payload(raw_checks)), _RawSynthesis,
            self._settings.llm_synthesis_max_response_tokens,
        )
        findings, questions = self._hydrate_synthesis(synthesis, raw_checks, catalog)
        return AccountAssessment(
            case_id=request.case_id, acct_num=request.acct_num, status="needs_review",
            decision="close_case" if synthesis.risk_level == "low" else "continue_due_diligence",
            risk_level=synthesis.risk_level, executive_summary=synthesis.executive_summary,
            review_checks=checks, findings=findings, reviewer_questions=questions,
            limitations=[
                AssessmentLimitation(limitation="Assessment uses only six monthly aggregates and linked customer-profile history."),
                AssessmentLimitation(limitation="No counterparty, transaction narrative, channel, transaction sequence, expected turnover, declared income, or source-of-funds data was available."),
                AssessmentLimitation(limitation="Citizenship is not used as a transaction-risk factor without geographical transaction or sanctions data."),
                AssessmentLimitation(limitation="LLM output is decision support and requires authorised human review."),
            ],
            months_reviewed=len(request.monthly_summary),
            profile_records_matched=len(request.customer_profile),
            generated_at=datetime.now(timezone.utc),
        )

    async def _ask_stage(
        self, case_id: str, stage: str, system_prompt: str, payload: dict[str, Any],
        model: type[ModelType], max_tokens: int,
    ) -> ModelType:
        raw = await self._llm_queue.submit(
            name=f"case={case_id} stage={stage}",
            operation=lambda: self._llm.complete_json(
                system_prompt=system_prompt, user_payload=payload,
                response_schema=model.model_json_schema(), schema_name=model.__name__,
                max_response_tokens=max_tokens,
            ),
        )
        try:
            return model.model_validate(raw)
        except ValidationError as exc:
            raise ModelOutputError(f"{stage} stage output did not meet the required schema: {exc}") from exc

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
    def _evidence_catalog(request: AccountAnalysisRequest, profile_payload: list[dict[str, str | None]]) -> dict[str, EvidenceItem]:
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
        for profile in profile_payload:
            for field, label in labels.items():
                if (value := profile.get(field)) is not None:
                    item_id = f"{profile['profile_record_id']}.{field}"
                    catalog[item_id] = EvidenceItem(evidence_id=item_id, source="customer_profile", label=f"{profile['profile_record_id']} · {label}", value=value)
        return catalog

    @staticmethod
    def _timeline_fields() -> set[str]:
        return {"txn_count_monthly", "total_amount", "avg_amount", "std_amount", "max_amount"}

    @staticmethod
    def _flow_fields() -> set[str]:
        return {"txn_count_monthly", "debit_count_monthly", "credit_count_monthly", "monthly_debit", "monthly_credit", "pct_burst", "pct_trx_gap"}

    @staticmethod
    def _monthly_ids(catalog: dict[str, EvidenceItem], fields: set[str]) -> list[str]:
        return [item_id for item_id in catalog if item_id.startswith("M") and item_id.rsplit(".", 1)[1] in fields]

    @staticmethod
    def _profile_stage_ids(catalog: dict[str, EvidenceItem]) -> list[str]:
        # Citizenship is deliberately non-citable in this transaction assessment.
        return [item_id for item_id in catalog if not item_id.endswith((".citizen_cd", ".citizenship"))]

    @staticmethod
    def _ordered_raw_checks(timeline: "_RawTimelineStage", flow: "_RawFlowStage", profile: "_RawProfileStage") -> dict[ReviewCheckName, "_RawReviewCheck"]:
        return {
            "dormancy_reactivation": timeline.dormancy_reactivation,
            "activity_value_change": timeline.activity_value_change,
            "debit_credit_flow": flow.debit_credit_flow,
            "burst_and_gaps": flow.burst_and_gaps,
            "profile_consistency": profile.profile_consistency,
        }

    @staticmethod
    def _validate_stage_evidence(stage: str, checks: tuple["_RawReviewCheck", ...], allowed_ids: list[str]) -> None:
        allowed = set(allowed_ids)
        invalid = sorted({item_id for check in checks for item_id in check.evidence_ids if item_id not in allowed})
        if invalid:
            raise ModelOutputError(f"{stage} stage cited evidence outside its supplied evidence set: {', '.join(invalid)}")

    @staticmethod
    def _resolve(ids: list[str], catalog: dict[str, EvidenceItem], label: str) -> list[EvidenceItem]:
        unknown = [item_id for item_id in ids if item_id not in catalog]
        if unknown:
            raise ModelOutputError(f"LLM cited evidence not present in supplied data for {label}: {', '.join(unknown)}")
        return [catalog[item_id] for item_id in dict.fromkeys(ids)]

    def _hydrate_checks(self, raw_checks: dict[ReviewCheckName, "_RawReviewCheck"], catalog: dict[str, EvidenceItem]) -> list[ReviewCheck]:
        checks: list[ReviewCheck] = []
        for name in _SIGNAL_ORDER:
            raw = raw_checks[name]
            if name != "profile_consistency" and not raw.evidence_ids:
                raise ModelOutputError(f"{name} requires at least one evidence ID")
            if name == "profile_consistency" and raw.outcome == "observed":
                has_monthly = any(item_id.startswith("M") for item_id in raw.evidence_ids)
                has_profile = any(item_id.startswith("P") for item_id in raw.evidence_ids)
                if not (has_monthly and has_profile):
                    raise ModelOutputError("observed profile consistency requires both monthly and profile evidence")
            checks.append(ReviewCheck(check=name, outcome=raw.outcome, rationale=raw.rationale, evidence=self._resolve(raw.evidence_ids, catalog, name)))
        return checks

    @staticmethod
    def _signal_payload(raw_checks: dict[ReviewCheckName, "_RawReviewCheck"]) -> list[dict[str, Any]]:
        return [{"signal_id": _SIGNAL_IDS[name], "check": name, "outcome": raw_checks[name].outcome, "rationale": raw_checks[name].rationale} for name in _SIGNAL_ORDER]

    def _hydrate_synthesis(self, raw: "_RawSynthesis", raw_checks: dict[ReviewCheckName, "_RawReviewCheck"], catalog: dict[str, EvidenceItem]) -> tuple[list[AccountFinding], list[ReviewerQuestion]]:
        by_id = {signal_id: name for name, signal_id in _SIGNAL_IDS.items()}
        names = [by_id[item.signal_id] for item in raw.selected_findings]
        if len(set(names)) != len(names):
            raise ModelOutputError("synthesis selected the same signal more than once")
        for name in names:
            if raw_checks[name].outcome != "observed":
                raise ModelOutputError(f"synthesis selected signal '{name}' that was not observed")
        if "profile_consistency" in names and not (set(names) & _TRANSACTION_SIGNALS):
            raise ModelOutputError("profile consistency cannot be a standalone material finding")
        findings = [AccountFinding(
            finding_id=str(uuid4()), category=_SIGNAL_CATEGORY[name], severity=item.severity,
            rationale=raw_checks[name].rationale, evidence=self._resolve(raw_checks[name].evidence_ids, catalog, name),
        ) for item, name in zip(raw.selected_findings, names, strict=True)]
        highest: RiskLevel = max((item.severity for item in raw.selected_findings), key=lambda value: _SEVERITY_RANK[value], default="low")
        if raw.risk_level != highest:
            raise ModelOutputError("risk_level must equal the highest selected finding severity, or low with no findings")
        if raw.risk_level == "high" and len(findings) < 2:
            raise ModelOutputError("high risk requires at least two corroborating material findings")
        questions: list[ReviewerQuestion] = []
        selected_ids = {_SIGNAL_IDS[name] for name in names}
        for item in raw.reviewer_questions:
            if not set(item.signal_ids) <= selected_ids:
                raise ModelOutputError("reviewer question referenced a non-selected signal")
            evidence_ids = [evidence_id for signal_id in item.signal_ids for evidence_id in raw_checks[by_id[signal_id]].evidence_ids]
            questions.append(ReviewerQuestion(question=item.question, evidence=self._resolve(evidence_ids, catalog, "reviewer question")))
        return findings, questions


class _RawReviewCheck(BaseModel):
    model_config = ConfigDict(extra="forbid")
    outcome: ReviewOutcome
    rationale: str = Field(min_length=1, max_length=220)
    evidence_ids: list[str] = Field(default_factory=list, max_length=3)


class _RawTimelineStage(BaseModel):
    model_config = ConfigDict(extra="forbid")
    dormancy_reactivation: _RawReviewCheck
    activity_value_change: _RawReviewCheck


class _RawFlowStage(BaseModel):
    model_config = ConfigDict(extra="forbid")
    debit_credit_flow: _RawReviewCheck
    burst_and_gaps: _RawReviewCheck


class _RawProfileStage(BaseModel):
    model_config = ConfigDict(extra="forbid")
    profile_consistency: _RawReviewCheck


class _RawSelectedFinding(BaseModel):
    model_config = ConfigDict(extra="forbid")
    signal_id: str = Field(pattern=r"^S[1-5]$")
    severity: FindingSeverity


class _RawReviewerQuestion(BaseModel):
    model_config = ConfigDict(extra="forbid")
    question: str = Field(min_length=1, max_length=240)
    signal_ids: list[str] = Field(min_length=1, max_length=2)


class _RawSynthesis(BaseModel):
    model_config = ConfigDict(extra="forbid")
    risk_level: RiskLevel
    executive_summary: str = Field(min_length=1, max_length=420)
    selected_findings: list[_RawSelectedFinding] = Field(default_factory=list, max_length=3)
    reviewer_questions: list[_RawReviewerQuestion] = Field(default_factory=list, max_length=2)

    @model_validator(mode="after")
    def decision_contract(self) -> "_RawSynthesis":
        selected = [item.signal_id for item in self.selected_findings]
        if self.risk_level == "low" and selected:
            raise ValueError("low risk requires no selected findings")
        if self.risk_level != "low" and not selected:
            raise ValueError("medium or high risk requires a selected finding")
        if len(set(selected)) != len(selected):
            raise ValueError("selected findings must be unique")
        for question in self.reviewer_questions:
            if not set(question.signal_ids) <= set(selected):
                raise ValueError("reviewer questions can reference only selected findings")
        return self
