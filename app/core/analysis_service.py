from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Literal, TypeVar
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from app.adapters.llm_client import LlmOutputFormatError, LlmOutputTruncatedError
from app.config import Settings
from app.core.llm_work_queue import LlmWorkQueue
from app.core.models import (
    AccountAnalysisRequest, AccountAssessment, AccountFinding, AssessmentLimitation,
    CustomerProfileContext, CustomerProfileRecord, EvidenceItem, FindingCategory, LlmClient,
    ReviewCheck, ReviewCheckName, ReviewOutcome, RiskLevel,
)
from app.core.prompts import (
    FORMAT_RETRY_SUFFIX, PROFILE_CONTEXT_SYSTEM_PROMPT, TRANSACTION_CONTEXT_SYSTEM_PROMPT,
    profile_context_input, transaction_context_input,
)
from app.core.reference_data import resolve_citizenship, resolve_occupation

logger = logging.getLogger("app.analysis_service")
ModelType = TypeVar("ModelType", bound=BaseModel)
TransactionOutcome = Literal["observed", "not_observed"]
_CHECK_ORDER: tuple[ReviewCheckName, ...] = (
    "dormancy_reactivation", "activity_value_change", "debit_credit_flow", "burst_and_gaps",
)
_TRANSACTION_FIELDS: dict[ReviewCheckName, frozenset[str]] = {
    "dormancy_reactivation": frozenset({"txn_count_monthly"}),
    "activity_value_change": frozenset({"txn_count_monthly", "total_amount", "avg_amount", "std_amount", "max_amount"}),
    "debit_credit_flow": frozenset({"debit_count_monthly", "credit_count_monthly", "monthly_debit", "monthly_credit", "monthly_avg_debit", "monthly_avg_credit"}),
    "burst_and_gaps": frozenset({"pct_burst", "pct_trx_gap", "txn_count_monthly"}),
}
_PROFILE_CONTEXT_FIELDS = frozenset({
    "occupation_cd", "occupation", "citizen_cd", "citizenship", "indv_org_type",
    "last_maint_dt", "valid_from_dttm", "valid_to_dttm",
})
_FORBIDDEN_TRANSACTION_TEXT = ("n/a", "insufficient", "nothing happened", "please provide")
_FORBIDDEN_PROFILE_TEXT = ("n/a", "insufficient", "missing", "unavailable", "not provided", "please provide")


class ModelOutputError(ValueError):
    """A model response was not complete, grounded, and safe to display."""


@dataclass(frozen=True)
class _FlatCheck:
    outcome: TransactionOutcome
    context: str
    evidence_text: str


class AnalysisService:
    """Concurrent transaction and factual-profile summaries with strict validation.

    The model determines transaction interpretation. The application only
    serialises source data, validates citations/contracts, and renders results.
    """

    def __init__(self, llm: LlmClient, settings: Settings, llm_queue: LlmWorkQueue) -> None:
        self._llm = llm
        self._settings = settings
        self._llm_queue = llm_queue

    async def analyze_account(self, request: AccountAnalysisRequest) -> AccountAssessment:
        profile = self._profile_records_for_llm(request.customer_profile)
        catalog = self._evidence_catalog(request, profile)
        monthly = [row.model_dump(mode="json") for row in request.monthly_summary]

        transaction_task = asyncio.create_task(self._complete_with_retry(
            request.case_id, "transaction-context", TRANSACTION_CONTEXT_SYSTEM_PROMPT,
            transaction_context_input(monthly), _RawTransactionContext,
            self._settings.llm_transaction_max_response_tokens,
            lambda value: self._validate_transaction_context(value, catalog),
        ))
        profile_task = asyncio.create_task(self._profile_context_or_none(request.case_id, profile, catalog))
        transaction_result, profile_result = await asyncio.gather(transaction_task, profile_task, return_exceptions=True)

        if isinstance(transaction_result, Exception):
            logger.error("Transaction context failed: case=%s error=%s", request.case_id, transaction_result)
            return self._manual_review_fallback(request)

        profile_context: CustomerProfileContext | None = None
        if isinstance(profile_result, Exception):
            logger.warning("Profile summary omitted: case=%s error=%s", request.case_id, profile_result)
        elif profile_result is not None:
            profile_context = self._hydrate_profile_context(profile_result, catalog)

        try:
            return self._hydrate_assessment(request, transaction_result, profile_context, catalog)
        except ModelOutputError as exc:
            logger.error("Transaction context could not be rendered: case=%s error=%s", request.case_id, exc)
            return self._manual_review_fallback(request)

    async def _profile_context_or_none(
        self, case_id: str, profile: list[dict[str, str | None]], catalog: dict[str, EvidenceItem],
    ) -> "_RawProfileContext | None":
        if not profile:
            return None
        return await self._complete_with_retry(
            case_id, "profile-context", PROFILE_CONTEXT_SYSTEM_PROMPT,
            profile_context_input(profile), _RawProfileContext,
            self._settings.llm_profile_context_max_response_tokens,
            lambda value: self._validate_profile_context(value, catalog),
        )

    async def _complete_with_retry(
        self, case_id: str, stage: str, system_prompt: str, prompt_input: str,
        model_type: type[ModelType], max_response_tokens: int, validator: Callable[[ModelType], None],
    ) -> ModelType:
        try:
            result = await self._ask(case_id, stage, system_prompt, prompt_input, model_type, max_response_tokens)
            validator(result)
            return result
        except LlmOutputTruncatedError as exc:
            raise ModelOutputError(f"{stage} reached its output limit; no same-budget retry was attempted.") from exc
        except (LlmOutputFormatError, ModelOutputError) as first_error:
            if not self._settings.llm_format_retry_enabled:
                raise ModelOutputError(f"{stage} output was rejected and format retry is disabled.") from first_error
            logger.warning("Context output rejected; retrying once: case=%s stage=%s error=%s", case_id, stage, first_error)
            try:
                result = await self._ask(
                    case_id, f"{stage}-format-retry", system_prompt + FORMAT_RETRY_SUFFIX,
                    prompt_input, model_type, max_response_tokens,
                )
                validator(result)
                return result
            except LlmOutputTruncatedError as exc:
                raise ModelOutputError(f"{stage} retry reached its output limit.") from exc
            except (LlmOutputFormatError, ModelOutputError) as second_error:
                raise ModelOutputError(f"{stage} did not produce a verifiable result after one format retry.") from second_error

    async def _ask(
        self, case_id: str, stage: str, system_prompt: str, prompt_input: str,
        model_type: type[ModelType], max_response_tokens: int,
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

    def _validate_transaction_context(self, raw: "_RawTransactionContext", catalog: dict[str, EvidenceItem]) -> None:
        checks = raw.checks()
        for name, check in checks.items():
            self._reject_forbidden_text(check.context, _FORBIDDEN_TRANSACTION_TEXT, f"{name} context")
            ids = self._split_evidence_ids(check.evidence_text, f"{name} evidence")
            self._validate_transaction_evidence(name, ids, catalog)
        observed_count = sum(check.outcome == "observed" for check in checks.values())
        if raw.risk_level == "low" and observed_count:
            raise ModelOutputError("low risk requires all transaction checks to be not_observed")
        if raw.risk_level == "medium" and not observed_count:
            raise ModelOutputError("medium risk requires an observed transaction check")
        if raw.risk_level == "high" and observed_count < 2:
            raise ModelOutputError("high risk requires two observed transaction checks")

    def _validate_profile_context(self, raw: "_RawProfileContext", catalog: dict[str, EvidenceItem]) -> None:
        self._reject_forbidden_text(raw.profile_summary, _FORBIDDEN_PROFILE_TEXT, "profile summary")
        ids = self._split_evidence_ids(raw.profile_evidence, "profile evidence")
        if any(not item_id.startswith("P") for item_id in ids):
            raise ModelOutputError("profile summary must cite profile evidence only")
        if any(item_id.rsplit(".", 1)[-1] not in _PROFILE_CONTEXT_FIELDS for item_id in ids):
            raise ModelOutputError("profile summary cited an unsupported profile field")
        self._resolve(ids, catalog, "profile summary")

    def _hydrate_assessment(
        self, request: AccountAnalysisRequest, raw: "_RawTransactionContext",
        profile_context: CustomerProfileContext | None, catalog: dict[str, EvidenceItem],
    ) -> AccountAssessment:
        checks = raw.checks()
        review_checks: list[ReviewCheck] = []
        findings: list[AccountFinding] = []
        for name in _CHECK_ORDER:
            check = checks[name]
            evidence = self._resolve(self._split_evidence_ids(check.evidence_text, f"{name} evidence"), catalog, name)
            review_checks.append(ReviewCheck(check=name, outcome=check.outcome, rationale=check.context, evidence=evidence))
            if check.outcome == "observed":
                findings.append(AccountFinding(
                    finding_id=str(uuid4()), category=name,
                    severity="high" if raw.risk_level == "high" else "medium",
                    rationale=check.context, evidence=evidence,
                ))
        return AccountAssessment(
            case_id=request.case_id, acct_num=request.acct_num, status="completed",
            decision="close_case" if raw.risk_level == "low" else "continue_due_diligence",
            risk_level=raw.risk_level, executive_summary=raw.executive_summary,
            review_checks=review_checks, findings=findings, reviewer_questions=[],
            customer_profile_context=profile_context, limitations=self._static_limitations(),
            months_reviewed=len(request.monthly_summary), profile_records_matched=len(request.customer_profile),
            generated_at=datetime.now(timezone.utc),
        )

    def _hydrate_profile_context(self, raw: "_RawProfileContext", catalog: dict[str, EvidenceItem]) -> CustomerProfileContext:
        ids = self._split_evidence_ids(raw.profile_evidence, "profile evidence")
        return CustomerProfileContext(summary=raw.profile_summary, evidence=self._resolve(ids, catalog, "profile summary"))

    @staticmethod
    def _reject_forbidden_text(value: str, phrases: tuple[str, ...], label: str) -> None:
        if any(phrase in value.casefold() for phrase in phrases):
            raise ModelOutputError(f"{label} contains a prohibited missing-data phrase")

    @staticmethod
    def _split_evidence_ids(value: str, label: str) -> list[str]:
        ids = value.split(",") if value else []
        if len(ids) != 2 or len(ids) != len(set(ids)) or any(not item or item != item.strip() for item in ids):
            raise ModelOutputError(f"{label} must contain exactly two distinct comma-separated IDs without spaces")
        return ids

    def _validate_transaction_evidence(self, name: ReviewCheckName, ids: list[str], catalog: dict[str, EvidenceItem]) -> None:
        if any(not item_id.startswith("M") or "." not in item_id for item_id in ids):
            raise ModelOutputError(f"{name} evidence must cite monthly IDs")
        if len({item_id.split(".", 1)[0] for item_id in ids}) != 2:
            raise ModelOutputError(f"{name} evidence must cite two different months")
        if any(item_id.rsplit(".", 1)[-1] not in _TRANSACTION_FIELDS[name] for item_id in ids):
            raise ModelOutputError(f"{name} evidence cited an unrelated field")
        self._resolve(ids, catalog, f"{name} evidence")

    @staticmethod
    def _resolve(ids: list[str], catalog: dict[str, EvidenceItem], label: str) -> list[EvidenceItem]:
        unknown = [item_id for item_id in ids if item_id not in catalog]
        if unknown:
            raise ModelOutputError(f"LLM cited evidence not present in supplied data for {label}: {', '.join(unknown)}")
        return [catalog[item_id] for item_id in ids]

    @staticmethod
    def _static_limitations() -> list[AssessmentLimitation]:
        return [
            AssessmentLimitation(limitation="Assessment uses only six monthly aggregates and factual linked customer-profile records."),
            AssessmentLimitation(limitation="LLM output is decision support and requires authorised human review."),
        ]

    def _manual_review_fallback(self, request: AccountAnalysisRequest) -> AccountAssessment:
        checks = [ReviewCheck(check=name, outcome="insufficient_data", rationale="Transaction context could not be verified; no conclusion is shown.") for name in _CHECK_ORDER]
        return AccountAssessment(
            case_id=request.case_id, acct_num=request.acct_num, status="needs_review",
            decision="continue_due_diligence", risk_level="medium",
            executive_summary="The LLM did not produce a verifiable transaction summary. Do not close this case on the basis of this result.",
            review_checks=checks, findings=[], reviewer_questions=[], customer_profile_context=None,
            limitations=self._static_limitations() + [AssessmentLimitation(limitation="Transaction-context LLM output verification failed; an authorised reviewer must assess the supplied data directly.")],
            months_reviewed=len(request.monthly_summary), profile_records_matched=len(request.customer_profile),
            generated_at=datetime.now(timezone.utc),
        )

    def _profile_records_for_llm(self, records: list[CustomerProfileRecord]) -> list[dict[str, str | None]]:
        profile: list[dict[str, str | None]] = []
        resolved_labels = 0
        for index, record in enumerate(records, start=1):
            occupation = resolve_occupation(record.occupation_cd, self._settings.occupation_code_path)
            if occupation and not occupation.startswith("Unmapped occupation code"):
                resolved_labels += 1
            profile.append({
                "profile_record_id": f"P{index}", "occupation_cd": record.occupation_cd, "occupation": occupation,
                "citizen_cd": record.citizen_cd, "citizenship": resolve_citizenship(record.citizen_cd),
                "indv_org_type": record.indv_org_type,
                "last_maint_dt": record.last_maint_dt.isoformat() if record.last_maint_dt else None,
                "valid_from_dttm": record.valid_from_dttm.isoformat(),
                "valid_to_dttm": record.valid_to_dttm.isoformat() if record.valid_to_dttm else None,
            })
        if profile:
            logger.info(
                "Profile prompt prepared: records=%d occupation_code_present=%d occupation_label_resolved=%d citizenship_present=%d type_present=%d effective_dates_present=%d",
                len(profile), sum(item["occupation_cd"] is not None for item in profile), resolved_labels,
                sum(item["citizenship"] is not None for item in profile), sum(item["indv_org_type"] is not None for item in profile),
                sum(item["valid_from_dttm"] is not None for item in profile),
            )
        return profile

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
    executive_summary: str = Field(min_length=1, max_length=260)
    dormancy_outcome: TransactionOutcome
    dormancy_context: str = Field(min_length=1, max_length=140)
    dormancy_evidence: str = Field(min_length=1, max_length=160)
    activity_value_outcome: TransactionOutcome
    activity_value_context: str = Field(min_length=1, max_length=140)
    activity_value_evidence: str = Field(min_length=1, max_length=160)
    debit_credit_outcome: TransactionOutcome
    debit_credit_context: str = Field(min_length=1, max_length=140)
    debit_credit_evidence: str = Field(min_length=1, max_length=160)
    burst_gap_outcome: TransactionOutcome
    burst_gap_context: str = Field(min_length=1, max_length=140)
    burst_gap_evidence: str = Field(min_length=1, max_length=160)

    def checks(self) -> dict[ReviewCheckName, _FlatCheck]:
        return {
            "dormancy_reactivation": _FlatCheck(self.dormancy_outcome, self.dormancy_context, self.dormancy_evidence),
            "activity_value_change": _FlatCheck(self.activity_value_outcome, self.activity_value_context, self.activity_value_evidence),
            "debit_credit_flow": _FlatCheck(self.debit_credit_outcome, self.debit_credit_context, self.debit_credit_evidence),
            "burst_and_gaps": _FlatCheck(self.burst_gap_outcome, self.burst_gap_context, self.burst_gap_evidence),
        }


class _RawProfileContext(BaseModel):
    model_config = ConfigDict(extra="forbid")
    profile_summary: str = Field(min_length=1, max_length=220)
    profile_evidence: str = Field(min_length=1, max_length=160)
