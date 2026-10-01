from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Callable, Literal, TypeVar
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from app.adapters.llm_client import LlmOutputFormatError, LlmOutputTruncatedError
from app.config import Settings
from app.core.llm_work_queue import LlmWorkQueue
from app.core.monthly_facts import (
    InactivityRun, activity_pair_context, activity_six_month_context, dormancy_rationale, find_inactivity_run,
    flow_pair_context, flow_six_month_context, no_inactivity_rationale,
)
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

logger = logging.getLogger("TransactionSummary.analysis_service")
ModelType = TypeVar("ModelType", bound=BaseModel)
TransactionOutcome = Literal["observed", "not_observed"]
_CHECK_ORDER: tuple[ReviewCheckName, ...] = (
    "dormancy_reactivation", "activity_value_change", "debit_credit_flow", "burst_and_gaps",
)
_FORBIDDEN_TRANSACTION_TEXT = ("n/a", "insufficient", "nothing happened", "please provide")
_FORBIDDEN_PROFILE_TEXT = (
    "n/a", "please provide", "named p",
)
_MONTH_TOKEN = re.compile(r"^M?(\d{6})$")
_NO_EVIDENCE_MONTHS = "none"
_NOT_OBSERVED_DEFAULT_CONTEXT = "No material pattern identified for this check."

# The model selects the relevant months. The application then attaches exact
# CSV values, so no model-generated field ID or value reaches bank staff.
_EVIDENCE_FIELDS: dict[ReviewCheckName, tuple[str, ...]] = {
    "dormancy_reactivation": ("txn_count_monthly",),
    "activity_value_change": ("txn_count_monthly", "total_amount"),
    "debit_credit_flow": ("debit_count_monthly", "credit_count_monthly", "monthly_debit", "monthly_credit"),
    "burst_and_gaps": ("pct_burst", "pct_trx_gap"),
}


class ModelOutputError(ValueError):
    """A model response was not complete, grounded, and safe to display."""


@dataclass(frozen=True)
class _FlatCheck:
    outcome: TransactionOutcome
    context: str
    months_text: str


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
        run = self._inactivity_run(monthly)

        transaction_task = asyncio.create_task(self._complete_with_retry(
            request.case_id, "transaction-context", TRANSACTION_CONTEXT_SYSTEM_PROMPT,
            transaction_context_input(monthly, run, self._settings.dormancy_min_zero_months),
            _RawTransactionContext, self._settings.llm_transaction_max_response_tokens,
            lambda value: self._validate_transaction_context(value, catalog, monthly, run, request.case_id),
        ))
        profile_task = asyncio.create_task(self._profile_context_or_none(request.case_id, profile, monthly))
        transaction_result, profile_result = await asyncio.gather(transaction_task, profile_task, return_exceptions=True)

        profile_context: CustomerProfileContext | None = None
        if isinstance(profile_result, Exception):
            logger.warning("Profile summary omitted: case=%s error=%s", request.case_id, profile_result)
        elif profile_result is not None:
            profile_context = self._hydrate_profile_context(profile_result, catalog, profile, monthly)

        if isinstance(transaction_result, Exception):
            logger.error("Transaction context failed: case=%s error=%s", request.case_id, transaction_result)
            return self._manual_review_fallback(request, profile_context)

        try:
            return self._hydrate_assessment(request, transaction_result, profile_context, catalog, run)
        except ModelOutputError as exc:
            logger.error("Transaction context could not be rendered: case=%s error=%s", request.case_id, exc)
            return self._manual_review_fallback(request, profile_context)

    async def _profile_context_or_none(
        self, case_id: str, profile: list[dict[str, str | None]], monthly: list[dict[str, Any]],
    ) -> "_RawProfileContext | None":
        if not profile:
            return None
        return await self._complete_with_retry(
            case_id, "profile-context", PROFILE_CONTEXT_SYSTEM_PROMPT,
            profile_context_input(profile, monthly), _RawProfileContext,
            self._settings.llm_profile_context_max_response_tokens,
            self._validate_profile_context,
        )

    async def _complete_with_retry(
        self, case_id: str, stage: str, system_prompt: str, prompt_input: str,
        model_type: type[ModelType], max_response_tokens: int,
        validator: Callable[[ModelType], ModelType | None],
    ) -> ModelType:
        """Ask, validate, retry once on a format problem.

        A validator may return a corrected copy of the result (for example after
        reconciling a field with source facts); returning None keeps the original.
        """
        try:
            result = await self._ask(case_id, stage, system_prompt, prompt_input, model_type, max_response_tokens)
            corrected = validator(result)
            return result if corrected is None else corrected
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
                corrected = validator(result)
                return result if corrected is None else corrected
            except LlmOutputTruncatedError as exc:
                raise ModelOutputError(f"{stage} retry reached its output limit.") from exc
            except (LlmOutputFormatError, ModelOutputError) as second_error:
                logger.warning("Retry output rejected: case=%s stage=%s error=%s", case_id, stage, second_error)
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
            normalise = getattr(model_type, "normalise_payload", None)
            return model_type.model_validate(normalise(result) if normalise else result)
        except ValidationError as exc:
            raise ModelOutputError(f"{stage} output did not meet the required schema: {exc}") from exc

    def _inactivity_run(self, monthly: list[dict[str, Any]]) -> InactivityRun | None:
        settings = self._settings
        return find_inactivity_run(
            monthly, min_zero_months=settings.dormancy_min_zero_months,
            single_reference=Decimal(str(settings.dormancy_review_single_amount)),
            month_total_reference=Decimal(str(settings.dormancy_review_month_total)),
        )

    def _reconcile_dormancy(
        self, raw: "_RawTransactionContext", run: InactivityRun | None, case_id: str,
    ) -> "_RawTransactionContext":
        """Make the dormancy check agree with the source rows.

        Whether the pattern exists is arithmetic (``find_inactivity_run``), so the
        model cannot miss or invent it. The model keeps the risk judgement; the only
        override of it is the floor below. Every change is logged for monitoring.
        """
        if not self._settings.dormancy_enforce_facts:
            return raw
        if run is None:
            if raw.dormancy_outcome != "observed":
                return raw
            logger.warning("Dormancy reconciled: case=%s model reported a pattern but none exists in the rows", case_id)
            return raw.model_copy(update={
                "dormancy_outcome": "not_observed", "dormancy_months": _NO_EVIDENCE_MONTHS, "dormancy_context": "",
            })
        if raw.dormancy_outcome != "observed":
            logger.warning(
                "Dormancy reconciled: case=%s model missed %d zero months before %s", case_id, run.zero_months, run.active_month,
            )
        reconciled = raw.model_copy(update={
            "dormancy_outcome": "observed", "dormancy_months": f"{run.zero_end},{run.active_month}",
            "dormancy_context": f"Inactive {run.zero_months} months before {run.active_month}.",
        })
        if reconciled.risk_level == "low" and not self._reactivation_only(reconciled.checks(), run):
            logger.warning(
                "Risk raised to medium: case=%s amount_vs_reference=%s with other observed checks", case_id, run.amount_band,
            )
            reconciled = reconciled.model_copy(update={"risk_level": "medium"})
        return reconciled

    @classmethod
    def _reactivation_only(cls, checks: dict[ReviewCheckName, _FlatCheck], run: InactivityRun | None) -> bool:
        """True when the only observed patterns are one small first month after inactivity.

        That is the dormancy check below the reference amount, plus any
        activity/flow check that merely compares the last inactive month with that
        same first active month. Such an account may be low risk while the pattern
        is still shown to staff.
        """
        if run is None or run.amount_band != "below" or checks["dormancy_reactivation"].outcome != "observed":
            return False
        same_event = {run.zero_end, run.active_month}
        for name, check in checks.items():
            if name == "dormancy_reactivation" or check.outcome != "observed":
                continue
            if name == "burst_and_gaps":
                return False
            try:
                months = cls._months_for_check(check, f"{name} months")
            except ModelOutputError:
                return False
            if not set(months) <= same_event:
                return False
        return True

    def _validate_transaction_context(
        self, raw: "_RawTransactionContext", catalog: dict[str, EvidenceItem], monthly: list[dict[str, Any]],
        run: InactivityRun | None = None, case_id: str = "",
    ) -> "_RawTransactionContext":
        raw = self._reconcile_dormancy(raw, run, case_id)
        checks = raw.checks()
        known_months = {item_id[1:7] for item_id in catalog if item_id.startswith("M")}
        by_month = {row["year_month"]: row for row in monthly}
        for name, check in checks.items():
            if check.outcome == "observed" and not check.context.strip():
                raise ModelOutputError(f"{name} is observed but has no context")
            self._reject_forbidden_text(check.context, _FORBIDDEN_TRANSACTION_TEXT, f"{name} context")
            months = self._months_for_check(check, f"{name} months")
            self._validate_transaction_months(months, known_months, name)
            if name in {"activity_value_change", "debit_credit_flow"} and months:
                if months != sorted(months):
                    raise ModelOutputError(f"{name} months must be chronological")
                first, second = (by_month[month] for month in months)
                fields = _EVIDENCE_FIELDS[name]
                unchanged = all(Decimal(str(first[field])) == Decimal(str(second[field])) for field in fields)
                persistent_one_sided = name == "debit_credit_flow" and (
                    (first["debit_count_monthly"] > 0 and second["debit_count_monthly"] > 0
                     and first["credit_count_monthly"] == second["credit_count_monthly"] == 0)
                    or (first["credit_count_monthly"] > 0 and second["credit_count_monthly"] > 0
                        and first["debit_count_monthly"] == second["debit_count_monthly"] == 0)
                )
                if unchanged and not persistent_one_sided:
                    raise ModelOutputError(f"{name} selected months show no change in the cited fields")
        # The model's executive_summary is never displayed (the summary is built from the
        # checks), so its wording is not validated; a negated phrase such as "No
        # observed reactivation" must not discard an otherwise valid response.
        observed_count = sum(check.outcome == "observed" for check in checks.values())
        if raw.risk_level == "low" and observed_count and not self._reactivation_only(checks, run):
            raise ModelOutputError("low risk requires all transaction checks to be not_observed")
        if raw.risk_level == "medium" and not observed_count:
            raise ModelOutputError("medium risk requires an observed transaction check")
        if raw.risk_level == "high" and observed_count < 2:
            raise ModelOutputError("high risk requires two observed transaction checks")
        return raw

    def _validate_profile_context(self, raw: "_RawProfileContext") -> None:
        self._reject_forbidden_text(raw.profile_summary, _FORBIDDEN_PROFILE_TEXT, "profile summary")

    def _dormancy_view(
        self, check: _FlatCheck, months: list[str], run: InactivityRun | None, catalog: dict[str, EvidenceItem],
    ) -> tuple[str, list[EvidenceItem]]:
        """Staff text and evidence for the dormancy check, from source facts when enforced."""
        if self._settings.dormancy_enforce_facts:
            if run is not None and check.outcome == "observed":
                ids = [
                    f"M{run.zero_start}.txn_count_monthly", f"M{run.zero_end}.txn_count_monthly",
                    f"M{run.active_month}.txn_count_monthly", f"M{run.active_month}.total_amount",
                    f"M{run.active_month}.max_amount", f"M{run.active_month}.monthly_debit",
                    f"M{run.active_month}.monthly_credit",
                ]
                return dormancy_rationale(run), self._resolve(list(dict.fromkeys(ids)), catalog, "dormancy_reactivation")
            return no_inactivity_rationale(self._settings.dormancy_min_zero_months), []
        return (
            check.context.strip() or _NOT_OBSERVED_DEFAULT_CONTEXT,
            self._evidence_for_months("dormancy_reactivation", months, catalog),
        )

    def _hydrate_assessment(
        self, request: AccountAnalysisRequest, raw: "_RawTransactionContext",
        profile_context: CustomerProfileContext | None, catalog: dict[str, EvidenceItem],
        run: InactivityRun | None = None,
    ) -> AccountAssessment:
        checks = raw.checks()
        monthly = [row.model_dump(mode="json") for row in request.monthly_summary]
        review_checks: list[ReviewCheck] = []
        findings: list[AccountFinding] = []
        for name in _CHECK_ORDER:
            check = checks[name]
            months = self._months_for_check(check, f"{name} months")
            evidence = self._evidence_for_months(name, months, catalog)
            if name == "activity_value_change":
                rationale = activity_pair_context(monthly, months) if months else activity_six_month_context(monthly)
            elif name == "debit_credit_flow":
                rationale = flow_pair_context(monthly, months) if months else flow_six_month_context(monthly)
            elif name == "dormancy_reactivation":
                rationale, evidence = self._dormancy_view(check, months, run, catalog)
            else:
                rationale = check.context.strip() or _NOT_OBSERVED_DEFAULT_CONTEXT
            review_checks.append(ReviewCheck(check=name, outcome=check.outcome, rationale=rationale, evidence=evidence))
            # A low-risk observed pattern (a small first month after inactivity) is shown as
            # a review check only, so it is not presented to staff as a medium finding.
            small_reactivation = name == "dormancy_reactivation" and run is not None and run.amount_band == "below"
            if check.outcome == "observed" and raw.risk_level != "low" and not small_reactivation:
                findings.append(AccountFinding(
                    finding_id=str(uuid4()), category=name,
                    severity="high" if raw.risk_level == "high" else "medium",
                    rationale=rationale, evidence=evidence,
                ))
        return AccountAssessment(
            case_id=request.case_id, acct_num=request.acct_num, status="completed",
            decision="close_case" if raw.risk_level == "low" else "continue_due_diligence",
            risk_level=raw.risk_level, executive_summary=self._checked_executive_summary(checks),
            review_checks=review_checks, findings=findings, reviewer_questions=[],
            customer_profile_context=profile_context, limitations=self._static_limitations(),
            months_reviewed=len(request.monthly_summary), profile_records_matched=len(request.customer_profile),
            generated_at=datetime.now(timezone.utc),
        )

    def _hydrate_profile_context(
        self, raw: "_RawProfileContext", catalog: dict[str, EvidenceItem],
        profile: list[dict[str, str | None]], monthly: list[dict[str, Any]],
    ) -> CustomerProfileContext:
        return CustomerProfileContext(
            summary=raw.profile_summary,
            evidence=self._canonical_profile_evidence(catalog, profile, monthly),
        )

    @staticmethod
    def _checked_executive_summary(checks: dict[ReviewCheckName, _FlatCheck]) -> str:
        labels = {
            "dormancy_reactivation": "activity after an inactive period",
            "activity_value_change": "activity/value change",
            "debit_credit_flow": "debit/credit flow",
            "burst_and_gaps": "burst/gap pattern",
        }
        observed = [labels[name] for name in _CHECK_ORDER if checks[name].outcome == "observed"]
        return (
            "Six-month review identified " + ", ".join(observed) + " for staff review."
            if observed else "No material transaction pattern was selected from the six monthly rows."
        )

    @staticmethod
    def _reject_forbidden_text(value: str, phrases: tuple[str, ...], label: str) -> None:
        if any(phrase in value.casefold() for phrase in phrases):
            raise ModelOutputError(f"{label} contains a prohibited missing-data phrase")

    @staticmethod
    def _split_months(value: str, label: str) -> list[str]:
        """Accept two or more source months, e.g. ``202607,202608`` or ``M202607_M202608``.

        This fixes punctuation only; it never derives a month or invents a value.
        """
        months: list[str] = []
        for token in re.split(r"[,_;\s]+", value.strip()):
            if not token:
                continue
            match = _MONTH_TOKEN.fullmatch(token)
            if match is None:
                raise ModelOutputError(f"{label} contains an invalid month: {token}")
            if match.group(1) not in months:
                months.append(match.group(1))
        if len(months) < 2:
            raise ModelOutputError(f"{label} must contain at least two distinct YYYYMM values")
        return months

    @classmethod
    def _months_for_check(cls, check: _FlatCheck, label: str) -> list[str]:
        """Permit no focused evidence only for a negative model conclusion."""
        if check.outcome == "not_observed" and check.months_text.strip().casefold() == _NO_EVIDENCE_MONTHS:
            return []
        return cls._split_months(check.months_text, label)

    @staticmethod
    def _validate_transaction_months(months: list[str], known_months: set[str], name: ReviewCheckName) -> None:
        unknown = [month for month in months if month not in known_months]
        if unknown:
            raise ModelOutputError(f"{name} selected month(s) not present in supplied data: {', '.join(unknown)}")

    def _evidence_for_months(
        self, name: ReviewCheckName, months: list[str], catalog: dict[str, EvidenceItem],
    ) -> list[EvidenceItem]:
        ids = [f"M{month}.{field}" for month in months for field in _EVIDENCE_FIELDS[name]]
        return self._resolve(ids, catalog, name)

    @staticmethod
    def _canonical_profile_evidence(
        catalog: dict[str, EvidenceItem], profile: list[dict[str, str | None]],
        monthly: list[dict[str, Any]],
    ) -> list[EvidenceItem]:
        """Attach exact profile and amount facts without model-generated citations."""
        ids: list[str] = []
        latest = profile[-1]
        latest_id = latest["profile_record_id"]
        for field in (
            "occupation", "citizenship", "indv_org_type",
            "last_maint_dt", "valid_from_dttm", "valid_to_dttm",
        ):
            ids.append(f"{latest_id}.{field}")
        peak = max(monthly, key=lambda row: Decimal(str(row["total_amount"])))
        peak_single = max(monthly, key=lambda row: Decimal(str(row["max_amount"])))
        ids.extend((
            f"M{peak['year_month']}.total_amount",
            f"M{peak['year_month']}.monthly_debit",
            f"M{peak['year_month']}.monthly_credit",
            f"M{peak_single['year_month']}.max_amount",
        ))
        transitions = list(zip(profile, profile[1:]))
        for field in ("occupation", "citizenship", "indv_org_type"):
            for previous, current in reversed(transitions):
                if previous.get(field) != current.get(field):
                    ids.extend((
                        f"{previous['profile_record_id']}.{field}",
                        f"{current['profile_record_id']}.{field}",
                        f"{current['profile_record_id']}.valid_from_dttm",
                    ))
                    break
        ids = list(dict.fromkeys(item_id for item_id in ids if item_id in catalog))[:16]
        if not ids:
            raise ModelOutputError("no factual profile evidence is available to render the profile summary")
        return [catalog[item_id] for item_id in ids]

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

    def _manual_review_fallback(
        self, request: AccountAnalysisRequest, profile_context: CustomerProfileContext | None = None,
    ) -> AccountAssessment:
        checks = [ReviewCheck(check=name, outcome="insufficient_data", rationale="Transaction context could not be verified; no conclusion is shown.") for name in _CHECK_ORDER]
        return AccountAssessment(
            case_id=request.case_id, acct_num=request.acct_num, status="needs_review",
            decision="continue_due_diligence", risk_level="medium",
            executive_summary="The LLM did not produce a verifiable transaction summary. Do not close this case on the basis of this result.",
            review_checks=checks, findings=[], reviewer_questions=[], customer_profile_context=profile_context,
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
    dormancy_context: str = Field(max_length=140)
    dormancy_months: str = Field(min_length=1, max_length=32)
    activity_value_outcome: TransactionOutcome
    activity_value_context: str = Field(max_length=140)
    activity_value_months: str = Field(min_length=1, max_length=32)
    debit_credit_outcome: TransactionOutcome
    debit_credit_context: str = Field(max_length=140)
    debit_credit_months: str = Field(min_length=1, max_length=32)
    burst_gap_outcome: TransactionOutcome
    burst_gap_context: str = Field(max_length=140)
    burst_gap_months: str = Field(min_length=1, max_length=32)

    @classmethod
    def normalise_payload(cls, payload: Any) -> Any:
        """Replace a missing or unrecognised risk_level with the one the policy implies.

        Under the stated policy the risk level follows from the observed checks, so a
        stray value such as ``not_observed`` is recoverable and must not discard a
        response. The derived level is conservative (never high); the validator
        applies every other risk rule afterwards.
        """
        if not isinstance(payload, dict):
            return payload
        risk = str(payload.get("risk_level", "")).strip().lower()
        if risk in {"low", "medium", "high"}:
            return {**payload, "risk_level": risk}
        observed = sum(payload.get(key) == "observed" for key in (
            "dormancy_outcome", "activity_value_outcome", "debit_credit_outcome", "burst_gap_outcome",
        ))
        derived = "medium" if observed else "low"
        logger.warning("risk_level %r is not low/medium/high; using %s from %d observed check(s)", risk, derived, observed)
        return {**payload, "risk_level": derived}

    def checks(self) -> dict[ReviewCheckName, _FlatCheck]:
        return {
            "dormancy_reactivation": _FlatCheck(self.dormancy_outcome, self.dormancy_context, self.dormancy_months),
            "activity_value_change": _FlatCheck(self.activity_value_outcome, self.activity_value_context, self.activity_value_months),
            "debit_credit_flow": _FlatCheck(self.debit_credit_outcome, self.debit_credit_context, self.debit_credit_months),
            "burst_and_gaps": _FlatCheck(self.burst_gap_outcome, self.burst_gap_context, self.burst_gap_months),
        }


class _RawProfileContext(BaseModel):
    model_config = ConfigDict(extra="forbid")
    profile_summary: str = Field(min_length=1, max_length=650)