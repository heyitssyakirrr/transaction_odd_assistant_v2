from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from typing import Any, Callable, ClassVar, Literal, TypeVar
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from app.adapters.llm_client import LlmOutputFormatError, LlmOutputTruncatedError
from app.config import Settings
from app.core.llm_work_queue import LlmWorkQueue
from app.core.monthly_facts import (
    InactivityRun, activity_pair_context, activity_six_month_context, allowed_burst_gap_numbers,
    burst_gap_pair_facts, burst_gap_six_month_context, dormancy_rationale, find_inactivity_run,
    flow_pair_context, flow_six_month_context, no_inactivity_rationale,
)
from app.core.models import (
    AccountAnalysisRequest, AccountAssessment, AccountFinding, AssessmentLimitation,
    CustomerProfileContext, CustomerProfileRecord, EvidenceItem, LlmClient,
    ReviewCheck, ReviewCheckName, RiskLevel,
)
from app.core.prompts import (
    ACTIVITY_FLOW_SYSTEM_PROMPT, FORMAT_RETRY_SUFFIX, PROFILE_CONTEXT_SYSTEM_PROMPT, TIMING_SYSTEM_PROMPT,
    activity_flow_input, profile_context_input, timing_input,
)
from app.core.reference_data import resolve_citizenship, resolve_occupation

logger = logging.getLogger("TransactionSummary.analysis_service")
ModelType = TypeVar("ModelType", bound=BaseModel)
TransactionOutcome = Literal["observed", "not_observed"]
_CHECK_ORDER: tuple[ReviewCheckName, ...] = (
    "dormancy_reactivation", "activity_value_change", "debit_credit_flow", "burst_and_gaps",
)
_FORBIDDEN_TRANSACTION_TEXT = ("n/a", "insufficient", "nothing happened", "please provide")
# Wording that would turn neutral timing context into an allegation.
_FORBIDDEN_INSIGHT_TEXT = _FORBIDDEN_TRANSACTION_TEXT + (
    "launder", "illicit", "illegal", "fraud", "criminal", "terroris", "smurf",
)
_FORBIDDEN_PROFILE_TEXT = (
    "n/a", "please provide", "named p",
)
_MONTH_TOKEN = re.compile(r"^M?(\d{6})$")
_NUMBER_TOKEN = re.compile(r"\d[\d,]*(?:\.\d+)?")
_NO_EVIDENCE_MONTHS = "none"
_RISK_ORDER: dict[RiskLevel, int] = {"low": 0, "medium": 1, "high": 2}
_RATIONALE_LIMIT = 360

# The model selects the relevant months. The application then attaches exact
# CSV values, so no model-generated field ID or value reaches bank staff.
_EVIDENCE_FIELDS: dict[ReviewCheckName, tuple[str, ...]] = {
    "dormancy_reactivation": ("txn_count_monthly",),
    "activity_value_change": ("txn_count_monthly", "total_amount"),
    "debit_credit_flow": ("debit_count_monthly", "credit_count_monthly", "monthly_debit", "monthly_credit"),
    "burst_and_gaps": ("txn_count_monthly", "pct_burst", "pct_trx_gap"),
}
_CHECK_LABELS: dict[ReviewCheckName, str] = {
    "dormancy_reactivation": "activity after an inactive period",
    "activity_value_change": "activity/value change",
    "debit_credit_flow": "debit/credit flow",
    "burst_and_gaps": "burst/gap pattern",
}


class ModelOutputError(ValueError):
    """A model response was not complete, grounded, and safe to display."""


@dataclass(frozen=True)
class _FlatCheck:
    outcome: TransactionOutcome
    context: str
    months_text: str


@dataclass(frozen=True)
class _TransactionCall:
    """One focused LLM call covering a fixed group of review checks.

    Adding a check group means adding one of these; groups never share a prompt,
    so a prompt change in one group cannot alter another group's answers.
    """

    stage: str
    checks: tuple[ReviewCheckName, ...]
    system_prompt: str
    prompt_input: str
    model_type: type["_RawCheckGroup"]
    validator: Callable[[Any], Any]


class AnalysisService:
    """Concurrent, focused LLM calls with strict validation.

    Three calls run in parallel through the shared queue: activity/flow, timing
    (dormancy and burst/gaps) and the customer profile. The model interprets the
    data and judges risk; the application prepares labelled facts, validates the
    answers against the source rows, and renders exact evidence. A failed call
    only affects its own checks.
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
        calls = self._transaction_calls(request.case_id, monthly, catalog, run)

        call_tasks = [
            asyncio.create_task(self._complete_with_retry(
                request.case_id, call.stage, call.system_prompt, call.prompt_input, call.model_type,
                self._settings.llm_transaction_max_response_tokens, call.validator,
            ))
            for call in calls
        ]
        profile_task = asyncio.create_task(self._profile_context_or_none(request.case_id, profile, monthly))
        *call_results, profile_result = await asyncio.gather(*call_tasks, profile_task, return_exceptions=True)

        profile_context: CustomerProfileContext | None = None
        if isinstance(profile_result, Exception):
            logger.warning("Profile summary omitted: case=%s error=%s", request.case_id, profile_result)
        elif profile_result is not None:
            profile_context = self._hydrate_profile_context(profile_result, catalog, profile, monthly)

        results: dict[str, Any] = {}
        for call, result in zip(calls, call_results):
            if isinstance(result, Exception):
                logger.error("Transaction call failed: case=%s stage=%s error=%s", request.case_id, call.stage, result)
            else:
                results[call.stage] = result
        if not results:
            return self._manual_review_fallback(request, profile_context)

        try:
            return self._hydrate_assessment(request, calls, results, profile_context, catalog, run)
        except ModelOutputError as exc:
            logger.error("Transaction context could not be rendered: case=%s error=%s", request.case_id, exc)
            return self._manual_review_fallback(request, profile_context)

    def _transaction_calls(
        self, case_id: str, monthly: list[dict[str, Any]], catalog: dict[str, EvidenceItem],
        run: InactivityRun | None,
    ) -> list[_TransactionCall]:
        return [
            _TransactionCall(
                stage="activity-flow-context",
                checks=("activity_value_change", "debit_credit_flow"),
                system_prompt=ACTIVITY_FLOW_SYSTEM_PROMPT,
                prompt_input=activity_flow_input(monthly),
                model_type=_RawActivityFlowContext,
                validator=lambda raw: self._validate_activity_flow(raw, catalog, monthly),
            ),
            _TransactionCall(
                stage="timing-context",
                checks=("dormancy_reactivation", "burst_and_gaps"),
                system_prompt=TIMING_SYSTEM_PROMPT,
                prompt_input=timing_input(monthly, run, self._settings.dormancy_min_zero_months),
                model_type=_RawTimingContext,
                validator=lambda raw: self._validate_timing(raw, catalog, monthly, run, case_id),
            ),
        ]

    # --- validation -----------------------------------------------------------

    def _validate_activity_flow(
        self, raw: "_RawActivityFlowContext", catalog: dict[str, EvidenceItem], monthly: list[dict[str, Any]],
    ) -> None:
        """Unchanged activity/flow checks: real, chronological months that show a change."""
        known_months = self._known_months(catalog)
        by_month = {row["year_month"]: row for row in monthly}
        for name, check in raw.checks().items():
            if check.outcome == "observed" and not check.context.strip():
                raise ModelOutputError(f"{name} is observed but has no context")
            self._reject_forbidden_text(check.context, _FORBIDDEN_TRANSACTION_TEXT, f"{name} context")
            months = self._months_for_check(check, f"{name} months")
            self._validate_transaction_months(months, known_months, name)
            if not months:
                continue
            if months != sorted(months):
                raise ModelOutputError(f"{name} months must be chronological")
            first, second = (by_month[month] for month in months[:2])
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

    def _validate_timing(
        self, raw: "_RawTimingContext", catalog: dict[str, EvidenceItem], monthly: list[dict[str, Any]],
        run: InactivityRun | None, case_id: str,
    ) -> "_RawTimingContext":
        """Check the timing answer; repair what can be repaired without the model.

        - Dormancy presence must match the computed INACTIVITY_RUN fact (arithmetic on
          the rows). A disagreement is corrected and logged; the model's risk stands.
        - Burst/gap months must be real and chronological (a retry is worthwhile).
        - The burst/gap insight may only quote numbers present in the rows. If it does
          not, it is dropped and factual text is shown instead; no retry is spent.
        """
        updates: dict[str, Any] = {}
        expected = "observed" if run is not None else "not_observed"
        if raw.dormancy_outcome != expected:
            logger.warning(
                "Dormancy reconciled: case=%s model=%s rows=%s", case_id, raw.dormancy_outcome, expected,
            )
            updates["dormancy_outcome"] = expected

        if raw.burst_gap_outcome == "observed":
            months = self._split_months(raw.burst_gap_months, "burst_and_gaps months", min_count=1, max_count=2)
            self._validate_transaction_months(months, self._known_months(catalog), "burst_and_gaps")
            if months != sorted(months):
                raise ModelOutputError("burst_and_gaps months must be chronological")
            updates["burst_gap_months"] = ",".join(months)
        else:
            updates["burst_gap_months"] = _NO_EVIDENCE_MONTHS

        problem = self._insight_problem(raw.burst_gap_insight, monthly)
        if problem:
            logger.warning("Burst/gap insight replaced with factual text: case=%s reason=%s", case_id, problem)
            updates["burst_gap_insight"] = ""
        return raw.model_copy(update=updates)

    @staticmethod
    def _insight_problem(text: str, monthly: list[dict[str, Any]]) -> str | None:
        """Why an LLM insight cannot be shown, or None when it is grounded and neutral."""
        if not text.strip():
            return "empty"
        lowered = text.casefold()
        if any(phrase in lowered for phrase in _FORBIDDEN_INSIGHT_TEXT):
            return "prohibited wording"
        allowed = allowed_burst_gap_numbers(monthly)
        for token in _NUMBER_TOKEN.findall(text):
            try:
                value = Decimal(token.replace(",", "")).normalize()
            except InvalidOperation:
                return f"unreadable number {token}"
            if value not in allowed:
                return f"number not in the rows: {token}"
        return None

    @staticmethod
    def _known_months(catalog: dict[str, EvidenceItem]) -> set[str]:
        return {item_id[1:7] for item_id in catalog if item_id.startswith("M")}

    # --- rendering ------------------------------------------------------------

    def _hydrate_assessment(
        self, request: AccountAnalysisRequest, calls: list[_TransactionCall], results: dict[str, Any],
        profile_context: CustomerProfileContext | None, catalog: dict[str, EvidenceItem],
        run: InactivityRun | None,
    ) -> AccountAssessment:
        monthly = [row.model_dump(mode="json") for row in request.monthly_summary]
        group_of = {name: call for call in calls for name in call.checks}
        review_checks: list[ReviewCheck] = []
        findings: list[AccountFinding] = []
        failed: list[ReviewCheckName] = []
        for name in _CHECK_ORDER:
            raw = results.get(group_of[name].stage)
            if raw is None:
                failed.append(name)
                review_checks.append(ReviewCheck(
                    check=name, outcome="insufficient_data",
                    rationale="This check's LLM output could not be verified; review the monthly rows directly.",
                ))
                continue
            outcome, rationale, evidence = self._render_check(name, raw, monthly, catalog, run)
            review_checks.append(ReviewCheck(check=name, outcome=outcome, rationale=rationale, evidence=evidence))
            # The model's own risk for this group decides whether an observed pattern is
            # a finding: a low-risk group shows its patterns as review checks only.
            if outcome == "observed" and raw.risk_level != "low" and evidence:
                findings.append(AccountFinding(
                    finding_id=str(uuid4()), category=name,
                    severity="high" if raw.risk_level == "high" else "medium",
                    rationale=rationale, evidence=evidence,
                ))

        # Overall risk is the highest group risk. A group that could not be verified
        # counts as medium so the case is never closed on partial evidence.
        risks = [results[call.stage].risk_level if call.stage in results else "medium" for call in calls]
        risk_level = max(risks, key=_RISK_ORDER.__getitem__)
        complete = not failed
        limitations = self._static_limitations()
        if failed:
            limitations.append(AssessmentLimitation(limitation=(
                "Not verified by the LLM: " + ", ".join(_CHECK_LABELS[name] for name in failed)
                + ". An authorised reviewer must assess these checks from the supplied data."
            )))
        return AccountAssessment(
            case_id=request.case_id, acct_num=request.acct_num,
            status="completed" if complete else "needs_review",
            decision="close_case" if complete and risk_level == "low" else "continue_due_diligence",
            risk_level=risk_level, executive_summary=self._checked_executive_summary(review_checks),
            review_checks=review_checks, findings=findings, reviewer_questions=[],
            customer_profile_context=profile_context, limitations=limitations,
            months_reviewed=len(request.monthly_summary), profile_records_matched=len(request.customer_profile),
            generated_at=datetime.now(timezone.utc),
        )

    def _render_check(
        self, name: ReviewCheckName, raw: Any, monthly: list[dict[str, Any]],
        catalog: dict[str, EvidenceItem], run: InactivityRun | None,
    ) -> tuple[TransactionOutcome, str, list[EvidenceItem]]:
        if name == "dormancy_reactivation":
            if raw.dormancy_outcome == "observed" and run is not None:
                ids = [
                    f"M{run.zero_start}.txn_count_monthly", f"M{run.zero_end}.txn_count_monthly",
                    f"M{run.active_month}.txn_count_monthly", f"M{run.active_month}.total_amount",
                    f"M{run.active_month}.max_amount", f"M{run.active_month}.monthly_debit",
                    f"M{run.active_month}.monthly_credit",
                ]
                evidence = self._resolve(list(dict.fromkeys(ids)), catalog, name)
                return "observed", dormancy_rationale(run), evidence
            return "not_observed", no_inactivity_rationale(self._settings.dormancy_min_zero_months), []

        if name == "burst_and_gaps":
            months = [] if raw.burst_gap_outcome != "observed" else raw.burst_gap_months.split(",")
            insight = raw.burst_gap_insight.strip()
            facts = burst_gap_pair_facts(monthly, months) if months else burst_gap_six_month_context(monthly)
            rationale = self._fit(insight, facts) if insight else (
                facts if not months else self._fit("Timing pattern selected for staff review.", facts)
            )
            return raw.burst_gap_outcome, rationale, self._evidence_for_months(name, months, catalog)

        check = raw.checks()[name]
        months = self._months_for_check(check, f"{name} months")
        if name == "activity_value_change":
            rationale = activity_pair_context(monthly, months) if months else activity_six_month_context(monthly)
        else:
            rationale = flow_pair_context(monthly, months) if months else flow_six_month_context(monthly)
        return check.outcome, rationale, self._evidence_for_months(name, months, catalog)

    @staticmethod
    def _fit(lead: str, facts: str) -> str:
        """Join an insight with its facts; keep the facts if the pair is too long."""
        joined = f"{lead} {facts}"
        return joined if len(joined) <= _RATIONALE_LIMIT else facts[:_RATIONALE_LIMIT]

    @staticmethod
    def _checked_executive_summary(review_checks: list[ReviewCheck]) -> str:
        observed = [_CHECK_LABELS[item.check] for item in review_checks if item.outcome == "observed"]
        unverified = [_CHECK_LABELS[item.check] for item in review_checks if item.outcome == "insufficient_data"]
        summary = (
            "Six-month review identified " + ", ".join(observed) + " for staff review."
            if observed else "No material transaction pattern was selected from the six monthly rows."
        )
        if unverified:
            summary += " Not verified: " + ", ".join(unverified) + "."
        return summary

    @staticmethod
    def _split_months(value: str, label: str, min_count: int = 2, max_count: int | None = None) -> list[str]:
        """Accept source months such as ``202607,202608`` or ``M202607_M202608``.

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
        if len(months) < min_count or (max_count is not None and len(months) > max_count):
            expected = f"{min_count}" if max_count == min_count else f"{min_count}-{max_count or 'n'}"
            raise ModelOutputError(f"{label} must contain {expected} distinct YYYYMM values")
        return months

    @classmethod
    def _months_for_check(cls, check: _FlatCheck, label: str) -> list[str]:
        """Permit no focused evidence only for a negative model conclusion."""
        if check.outcome == "not_observed" and check.months_text.strip().casefold() == _NO_EVIDENCE_MONTHS:
            return []
        return cls._split_months(check.months_text, label)

    def _evidence_for_months(
        self, name: ReviewCheckName, months: list[str], catalog: dict[str, EvidenceItem],
    ) -> list[EvidenceItem]:
        ids = [f"M{month}.{field}" for month in months for field in _EVIDENCE_FIELDS[name]]
        return self._resolve(ids, catalog, name)

    # --- unchanged helpers (LLM plumbing, profile, evidence) -------------------

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

    def _validate_profile_context(self, raw: "_RawProfileContext") -> None:
        self._reject_forbidden_text(raw.profile_summary, _FORBIDDEN_PROFILE_TEXT, "profile summary")

    def _hydrate_profile_context(
        self, raw: "_RawProfileContext", catalog: dict[str, EvidenceItem],
        profile: list[dict[str, str | None]], monthly: list[dict[str, Any]],
    ) -> CustomerProfileContext:
        return CustomerProfileContext(
            summary=raw.profile_summary,
            evidence=self._canonical_profile_evidence(catalog, profile, monthly),
        )

    @staticmethod
    def _reject_forbidden_text(value: str, phrases: tuple[str, ...], label: str) -> None:
        if any(phrase in value.casefold() for phrase in phrases):
            raise ModelOutputError(f"{label} contains a prohibited missing-data phrase")

    @staticmethod
    def _validate_transaction_months(months: list[str], known_months: set[str], name: ReviewCheckName) -> None:
        unknown = [month for month in months if month not in known_months]
        if unknown:
            raise ModelOutputError(f"{name} selected month(s) not present in supplied data: {', '.join(unknown)}")

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


class _RawCheckGroup(BaseModel):
    """Shared behaviour for the flat JSON answer of one check-group call."""

    model_config = ConfigDict(extra="forbid")
    risk_level: RiskLevel
    # Outcome keys of this group, used to repair an invalid risk_level.
    _OUTCOME_KEYS: ClassVar[tuple[str, ...]] = ()
    # Free-text keys that are capped rather than rejected when too long.
    _TEXT_LIMITS: ClassVar[dict[str, int]] = {}

    @classmethod
    def normalise_payload(cls, payload: Any) -> Any:
        """Repair recoverable answers instead of spending a retry on them.

        - A missing or unrecognised risk_level (e.g. ``not_observed``) becomes the
          level the prompt's own policy implies from the observed checks.
        - Text over its limit is cut at the limit. Activity/flow context is never
          displayed (that text is built from the rows), and the timing insight is
          re-checked before display, so a cut costs nothing.
        """
        if not isinstance(payload, dict):
            return payload
        payload = dict(payload)
        risk = str(payload.get("risk_level", "")).strip().lower()
        if risk in _RISK_ORDER:
            payload["risk_level"] = risk
        else:
            observed = sum(payload.get(key) == "observed" for key in cls._OUTCOME_KEYS)
            derived = "medium" if observed else "low"
            logger.warning("risk_level %r is not low/medium/high; using %s from %d observed check(s)", risk, derived, observed)
            payload["risk_level"] = derived
        for key, limit in cls._TEXT_LIMITS.items():
            value = payload.get(key)
            if isinstance(value, str) and len(value) > limit:
                payload[key] = value[:limit].rstrip()
        return payload


class _RawActivityFlowContext(_RawCheckGroup):
    _OUTCOME_KEYS: ClassVar[tuple[str, ...]] = ("activity_value_outcome", "debit_credit_outcome")
    _TEXT_LIMITS: ClassVar[dict[str, int]] = {"activity_value_context": 140, "debit_credit_context": 140}
    activity_value_outcome: TransactionOutcome
    activity_value_context: str = Field(max_length=140)
    activity_value_months: str = Field(min_length=1, max_length=32)
    debit_credit_outcome: TransactionOutcome
    debit_credit_context: str = Field(max_length=140)
    debit_credit_months: str = Field(min_length=1, max_length=32)

    def checks(self) -> dict[ReviewCheckName, _FlatCheck]:
        return {
            "activity_value_change": _FlatCheck(self.activity_value_outcome, self.activity_value_context, self.activity_value_months),
            "debit_credit_flow": _FlatCheck(self.debit_credit_outcome, self.debit_credit_context, self.debit_credit_months),
        }


class _RawTimingContext(_RawCheckGroup):
    _OUTCOME_KEYS: ClassVar[tuple[str, ...]] = ("dormancy_outcome", "burst_gap_outcome")
    # A cut insight usually loses its last number or clause; the grounding check
    # then decides whether it can still be shown.
    _TEXT_LIMITS: ClassVar[dict[str, int]] = {"burst_gap_insight": 240}
    dormancy_outcome: TransactionOutcome
    burst_gap_outcome: TransactionOutcome
    burst_gap_months: str = Field(min_length=1, max_length=32)
    burst_gap_insight: str = Field(default="", max_length=240)


class _RawProfileContext(BaseModel):
    model_config = ConfigDict(extra="forbid")
    profile_summary: str = Field(min_length=1, max_length=650)