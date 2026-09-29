from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, TypeVar
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from app.adapters.llm_client import LlmOutputFormatError, LlmOutputTruncatedError
from app.config import Settings
from app.core.llm_work_queue import LlmWorkQueue
from app.core.models import (
    AccountAnalysisRequest, AccountAssessment, AccountFinding, AssessmentLimitation,
    CustomerProfileRecord, EvidenceItem, FindingCategory, LlmClient,
    ReviewCheck, ReviewCheckName, ReviewOutcome, ReviewerQuestion, RiskLevel,
)
from app.core.prompts import (
    FORMAT_RETRY_SUFFIX, PROFILE_CONTEXT_SYSTEM_PROMPT, TRANSACTION_CONTEXT_SYSTEM_PROMPT,
    profile_context_input, transaction_context_input,
)
from app.core.reference_data import resolve_citizenship, resolve_occupation

logger = logging.getLogger("app.analysis_service")
ModelType = TypeVar("ModelType", bound=BaseModel)
_TRANSACTION_CHECKS: tuple[ReviewCheckName, ...] = (
    "dormancy_reactivation", "activity_value_change", "debit_credit_flow", "burst_and_gaps",
)
_CHECK_ORDER: tuple[ReviewCheckName, ...] = _TRANSACTION_CHECKS + ("profile_consistency",)
_TRANSACTION_FIELDS: dict[ReviewCheckName, frozenset[str]] = {
    "dormancy_reactivation": frozenset({"txn_count_monthly"}),
    "activity_value_change": frozenset({
        "txn_count_monthly", "total_amount", "avg_amount", "std_amount", "max_amount",
    }),
    "debit_credit_flow": frozenset({
        "debit_count_monthly", "credit_count_monthly", "monthly_debit", "monthly_credit",
        "monthly_avg_debit", "monthly_avg_credit",
    }),
    "burst_and_gaps": frozenset({"pct_burst", "pct_trx_gap", "txn_count_monthly"}),
}
_PROFILE_CONTEXT_FIELDS = frozenset({
    "occupation_cd", "occupation", "indv_org_type", "last_maint_dt", "valid_from_dttm", "valid_to_dttm",
})


class ModelOutputError(ValueError):
    """A model response was not a complete, grounded assessment."""


@dataclass(frozen=True)
class _FlatCheck:
    outcome: ReviewOutcome
    context: str
    evidence_text: str


class AnalysisService:
    """Two small, independent LLM context calls with strict source validation.

    The application orchestrates, validates, and renders model-written context.
    It does not calculate a risk score or create a transaction conclusion.
    """

    def __init__(self, llm: LlmClient, settings: Settings, llm_queue: LlmWorkQueue) -> None:
        self._llm = llm
        self._settings = settings
        self._llm_queue = llm_queue

    async def analyze_account(self, request: AccountAnalysisRequest) -> AccountAssessment:
        profile = self._profile_records_for_llm(request.customer_profile)
        catalog = self._evidence_catalog(request, profile)
        monthly = [row.model_dump(mode="json") for row in request.monthly_summary]

        transaction_task = asyncio.create_task(
            self._complete_with_format_retry(
                request.case_id, "transaction-context", TRANSACTION_CONTEXT_SYSTEM_PROMPT,
                transaction_context_input(monthly), _RawTransactionContext,
                self._settings.llm_transaction_max_response_tokens,
            )
        )
        profile_task = asyncio.create_task(
            self._profile_context_or_default(request.case_id, monthly, profile)
        )
        transaction_result, profile_result = await asyncio.gather(
            transaction_task, profile_task, return_exceptions=True,
        )

        if isinstance(transaction_result, Exception):
            logger.error("Transaction context failed: case=%s error=%s", request.case_id, transaction_result)
            return self._manual_review_fallback(request, str(transaction_result))

        profile_raw: _RawProfileContext | None
        profile_error: str | None = None
        if isinstance(profile_result, Exception):
            profile_raw = None
            profile_error = str(profile_result)
            logger.warning("Profile context unavailable: case=%s error=%s", request.case_id, profile_error)
        else:
            profile_raw = profile_result

        try:
            return self._hydrate_assessment(request, transaction_result, profile_raw, profile_error, catalog)
        except ModelOutputError as exc:
            logger.error("Validated transaction context could not be rendered: case=%s error=%s", request.case_id, exc)
            return self._manual_review_fallback(request, str(exc))

    async def _profile_context_or_default(
        self, case_id: str, monthly: list[dict[str, Any]], profile: list[dict[str, str | None]],
    ) -> "_RawProfileContext | None":
        if not profile:
            return None
        return await self._complete_with_format_retry(
            case_id, "profile-context", PROFILE_CONTEXT_SYSTEM_PROMPT,
            profile_context_input(monthly, profile), _RawProfileContext,
            self._settings.llm_profile_context_max_response_tokens,
        )

    async def _complete_with_format_retry(
        self,
        case_id: str,
        stage: str,
        system_prompt: str,
        prompt_input: str,
        model_type: type[ModelType],
        max_response_tokens: int,
    ) -> ModelType:
        try:
            return await self._ask(
                case_id, stage, system_prompt, prompt_input, model_type, max_response_tokens,
            )
        except LlmOutputTruncatedError as exc:
            raise ModelOutputError(f"{stage} reached its output limit; no same-budget retry was attempted.") from exc
        except (LlmOutputFormatError, ModelOutputError) as first_error:
            if not self._settings.llm_format_retry_enabled:
                raise ModelOutputError(f"{stage} output was rejected and format retry is disabled.") from first_error
            logger.warning("Context output rejected; retrying once: case=%s stage=%s error=%s", case_id, stage, first_error)
            try:
                return await self._ask(
                    case_id, f"{stage}-format-retry", system_prompt + FORMAT_RETRY_SUFFIX,
                    prompt_input, model_type, max_response_tokens,
                )
            except LlmOutputTruncatedError as exc:
                raise ModelOutputError(f"{stage} retry reached its output limit.") from exc
            except (LlmOutputFormatError, ModelOutputError) as second_error:
                raise ModelOutputError(f"{stage} did not produce a verifiable result after one format retry.") from second_error

    async def _ask(
        self,
        case_id: str,
        stage: str,
        system_prompt: str,
        prompt_input: str,
        model_type: type[ModelType],
        max_response_tokens: int,
    ) -> ModelType:
        result = await self._llm_queue.submit(
            name=f"case={case_id} stage={stage}",
            operation=lambda: self._llm.complete_json(
                system_prompt=system_prompt, user_payload=prompt_input,
                response_schema=model_type.model_json_schema(), schema_name=stage.replace("-", "_"),
                max_response_tokens=max_response_tokens,
            ),
        )
        try:
            return model_type.model_validate(result)
        except ValidationError as exc:
            raise ModelOutputError(f"{stage} output did not meet the required schema: {exc}") from exc

    def _hydrate_assessment(
        self,
        request: AccountAnalysisRequest,
        transaction: "_RawTransactionContext",
        profile: "_RawProfileContext | None",
        profile_error: str | None,
        catalog: dict[str, EvidenceItem],
    ) -> AccountAssessment:
        flat_checks = transaction.checks()
        checks: list[ReviewCheck] = []
        resolved_by_check: dict[ReviewCheckName, list[EvidenceItem]] = {}
        for name, flat in flat_checks.items():
            ids = self._split_evidence_ids(flat.evidence_text, f"{name} evidence", required=True)
            self._validate_transaction_evidence(name, ids, catalog)
            evidence = self._resolve(ids, catalog, f"review check '{name}'")
            resolved_by_check[name] = evidence
            checks.append(ReviewCheck(check=name, outcome=flat.outcome, rationale=flat.context, evidence=evidence))

        profile_check, profile_limitations = self._profile_check(profile, profile_error, catalog)
        checks.append(profile_check)
        priorities = self._priority_categories(transaction, flat_checks)
        findings = [
            AccountFinding(
                finding_id=str(uuid4()), category=category,
                severity=transaction.risk_level, rationale=flat_checks[category].context,
                evidence=resolved_by_check[category],
            )
            for category in priorities
        ]
        questions = self._reviewer_questions(transaction, catalog)
        return AccountAssessment(
            case_id=request.case_id, acct_num=request.acct_num, status="completed",
            decision="close_case" if transaction.risk_level == "low" else "continue_due_diligence",
            risk_level=transaction.risk_level, executive_summary=transaction.executive_summary,
            review_checks=checks, findings=findings, reviewer_questions=questions,
            limitations=self._static_limitations() + profile_limitations,
            months_reviewed=len(request.monthly_summary), profile_records_matched=len(request.customer_profile),
            generated_at=datetime.now(timezone.utc),
        )

    def _profile_check(
        self,
        profile: "_RawProfileContext | None", profile_error: str | None, catalog: dict[str, EvidenceItem],
    ) -> tuple[ReviewCheck, list[AssessmentLimitation]]:
        if profile is None:
            rationale = "No linked customer-profile record was supplied." if not profile_error else "Profile context could not be verified; no profile conclusion is shown."
            limitation = "No linked customer-profile record was available." if not profile_error else "Profile-context LLM output could not be verified; profile context was omitted."
            return (
                ReviewCheck(check="profile_consistency", outcome="insufficient_data", rationale=rationale),
                [AssessmentLimitation(limitation=limitation)],
            )
        ids = self._split_evidence_ids(
            profile.profile_evidence, "profile evidence", required=profile.profile_outcome != "insufficient_data",
        )
        if profile.profile_outcome != "insufficient_data":
            if sum(item_id.startswith("M") for item_id in ids) != 1 or sum(item_id.startswith("P") for item_id in ids) != 1:
                raise ModelOutputError("profile context requires one monthly and one profile citation")
        for item_id in ids:
            if item_id.startswith("P") and item_id.rsplit(".", 1)[-1] not in _PROFILE_CONTEXT_FIELDS:
                raise ModelOutputError("profile context cannot cite citizenship")
        evidence = self._resolve(ids, catalog, "profile context")
        return ReviewCheck(
            check="profile_consistency", outcome=profile.profile_outcome,
            rationale=profile.profile_context, evidence=evidence,
        ), []

    def _priority_categories(
        self, transaction: "_RawTransactionContext", checks: dict[ReviewCheckName, _FlatCheck],
    ) -> list[FindingCategory]:
        values = self._split_csv(transaction.priority_categories, "priority_categories")
        if len(values) > 2 or len(values) != len(set(values)):
            raise ModelOutputError("priority_categories must contain at most two distinct values")
        allowed = set(_TRANSACTION_CHECKS)
        if any(value not in allowed for value in values):
            raise ModelOutputError("priority_categories contains an invalid transaction category")
        if transaction.risk_level == "low" and values:
            raise ModelOutputError("low risk cannot contain priority categories")
        if transaction.risk_level == "medium" and not values:
            raise ModelOutputError("medium risk requires a material priority category")
        if transaction.risk_level == "high" and len(values) != 2:
            raise ModelOutputError("high risk requires two corroborating priority categories")
        if any(checks[value].outcome != "observed" for value in values):
            raise ModelOutputError("each priority category must have an observed context")
        return values  # type: ignore[return-value]

    def _reviewer_questions(
        self, transaction: "_RawTransactionContext", catalog: dict[str, EvidenceItem],
    ) -> list[ReviewerQuestion]:
        if not transaction.reviewer_question:
            if transaction.question_evidence:
                raise ModelOutputError("question_evidence must be empty when reviewer_question is empty")
            return []
        ids = self._split_evidence_ids(transaction.question_evidence, "question evidence", required=True)
        self._validate_monthly_ids(ids, "reviewer question")
        return [ReviewerQuestion(question=transaction.reviewer_question, evidence=self._resolve(ids, catalog, "reviewer question"))]

    @staticmethod
    def _split_csv(value: str, label: str) -> list[str]:
        if not value:
            return []
        values = value.split(",")
        if any(not item or item != item.strip() for item in values):
            raise ModelOutputError(f"{label} must be comma-separated without spaces")
        return values

    def _split_evidence_ids(self, value: str, label: str, *, required: bool) -> list[str]:
        ids = self._split_csv(value, label)
        if required and len(ids) != 2:
            raise ModelOutputError(f"{label} must contain exactly two evidence IDs")
        if not required and ids:
            raise ModelOutputError(f"{label} must be empty when not required")
        if len(ids) != len(set(ids)):
            raise ModelOutputError(f"{label} contains duplicate evidence IDs")
        return ids

    def _validate_transaction_evidence(
        self, name: ReviewCheckName, ids: list[str], catalog: dict[str, EvidenceItem],
    ) -> None:
        self._validate_monthly_ids(ids, f"{name} evidence")
        if len({item_id.split(".", 1)[0] for item_id in ids}) != 2:
            raise ModelOutputError(f"{name} evidence must cite two different months")
        if any(item_id.rsplit(".", 1)[-1] not in _TRANSACTION_FIELDS[name] for item_id in ids):
            raise ModelOutputError(f"{name} evidence cited an unrelated field")
        if any(item_id not in catalog for item_id in ids):
            raise ModelOutputError(f"{name} evidence cited unavailable data")

    @staticmethod
    def _validate_monthly_ids(ids: list[str], label: str) -> None:
        if any(not item_id.startswith("M") or "." not in item_id for item_id in ids):
            raise ModelOutputError(f"{label} must cite monthly evidence IDs")

    @staticmethod
    def _resolve(ids: list[str], catalog: dict[str, EvidenceItem], label: str) -> list[EvidenceItem]:
        unknown = [item_id for item_id in ids if item_id not in catalog]
        if unknown:
            raise ModelOutputError(f"LLM cited evidence not present in supplied data for {label}: {', '.join(unknown)}")
        return [catalog[item_id] for item_id in ids]

    @staticmethod
    def _static_limitations() -> list[AssessmentLimitation]:
        return [
            AssessmentLimitation(limitation="Assessment uses only six monthly aggregates and linked customer-profile history."),
            AssessmentLimitation(limitation="No counterparty, transaction narrative, channel, transaction sequence, expected turnover, declared income, or source-of-funds data was available."),
            AssessmentLimitation(limitation="Citizenship is not used as a transaction-risk factor without geographical transaction or sanctions data."),
            AssessmentLimitation(limitation="LLM output is decision support and requires authorised human review."),
        ]

    def _manual_review_fallback(self, request: AccountAnalysisRequest, _error: str) -> AccountAssessment:
        checks = [
            ReviewCheck(check=name, outcome="insufficient_data", rationale="Transaction context could not be verified; no conclusion is shown.")
            for name in _CHECK_ORDER
        ]
        return AccountAssessment(
            case_id=request.case_id, acct_num=request.acct_num, status="needs_review",
            decision="continue_due_diligence", risk_level="medium",
            executive_summary="The LLM did not produce a verifiable transaction assessment. Do not close this case on the basis of this result.",
            review_checks=checks, findings=[], reviewer_questions=[],
            limitations=self._static_limitations() + [AssessmentLimitation(limitation="Transaction-context LLM output verification failed; an authorised reviewer must assess the supplied data directly.")],
            months_reviewed=len(request.monthly_summary), profile_records_matched=len(request.customer_profile),
            generated_at=datetime.now(timezone.utc),
        )

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


class _RawTransactionContext(BaseModel):
    model_config = ConfigDict(extra="forbid")

    risk_level: RiskLevel
    executive_summary: str = Field(min_length=1, max_length=280)
    dormancy_outcome: ReviewOutcome
    dormancy_context: str = Field(min_length=1, max_length=140)
    dormancy_evidence: str = Field(min_length=1, max_length=160)
    activity_value_outcome: ReviewOutcome
    activity_value_context: str = Field(min_length=1, max_length=140)
    activity_value_evidence: str = Field(min_length=1, max_length=160)
    debit_credit_outcome: ReviewOutcome
    debit_credit_context: str = Field(min_length=1, max_length=140)
    debit_credit_evidence: str = Field(min_length=1, max_length=160)
    burst_gap_outcome: ReviewOutcome
    burst_gap_context: str = Field(min_length=1, max_length=140)
    burst_gap_evidence: str = Field(min_length=1, max_length=160)
    priority_categories: str = Field(max_length=100)
    reviewer_question: str = Field(max_length=200)
    question_evidence: str = Field(max_length=160)

    def checks(self) -> dict[ReviewCheckName, _FlatCheck]:
        return {
            "dormancy_reactivation": _FlatCheck(self.dormancy_outcome, self.dormancy_context, self.dormancy_evidence),
            "activity_value_change": _FlatCheck(self.activity_value_outcome, self.activity_value_context, self.activity_value_evidence),
            "debit_credit_flow": _FlatCheck(self.debit_credit_outcome, self.debit_credit_context, self.debit_credit_evidence),
            "burst_and_gaps": _FlatCheck(self.burst_gap_outcome, self.burst_gap_context, self.burst_gap_evidence),
        }


class _RawProfileContext(BaseModel):
    model_config = ConfigDict(extra="forbid")

    profile_outcome: ReviewOutcome
    profile_context: str = Field(min_length=1, max_length=160)
    profile_evidence: str = Field(max_length=160)
