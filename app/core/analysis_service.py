from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Callable, ClassVar, Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from app.adapters.llm_client import LlmOutputFormatError, LlmOutputTruncatedError
from app.config import Settings
from app.core.llm_work_queue import LlmWorkQueue
from app.core.monthly_facts import (
    InactivityRun, activity_pair_context, activity_six_month_context, allowed_burst_gap_numbers,
    burst_gap_evidence_months, burst_gap_pair_facts, burst_gap_six_month_context, check_table,
    dormancy_rationale, find_inactivity_run, flow_pair_context, flow_six_month_context, has_burst,
    has_in_month_gap, no_inactivity_rationale, numbers_in_text, pattern_problem, quotable_numbers,
    unquotable_number,
)
from app.core.models import (
    AccountAnalysisRequest, AccountAssessment, AssessmentLimitation, CustomerProfileContext,
    CustomerProfileRecord, Decision, EvidenceItem, EvidenceTable, LlmClient, OverallSummary,
    ReviewCheck, ReviewCheckName, RiskLevel,
)
from app.core.prompts import (
    ACTIVITY_MONEY_SYSTEM_PROMPT, FORMAT_RETRY_SUFFIX, INACTIVITY_BURST_GAPS_SYSTEM_PROMPT,
    OVERALL_SUMMARY_SYSTEM_PROMPT, PROFILE_CONTEXT_SYSTEM_PROMPT, activity_money_input,
    inactivity_burst_gaps_input, overall_summary_input, profile_context_input,
)
from app.core.reference_data import resolve_citizenship, resolve_occupation

logger = logging.getLogger("TransactionSummary.analysis_service")
ModelType = TypeVar("ModelType", bound=BaseModel)
Outcome = Literal["pattern_found", "no_pattern_found"]

# --- The four review checks ------------------------------------------------------
_CHECK_ORDER: tuple[ReviewCheckName, ...] = (
    "activity_after_inactivity", "activity_and_amount_change", "money_in_and_out", "burst_and_gaps",
)
_CHECK_TITLES: dict[ReviewCheckName, str] = {
    "activity_after_inactivity": "Activity after inactivity",
    "activity_and_amount_change": "Change in activity and amounts",
    "money_in_and_out": "Money in and money out",
    "burst_and_gaps": "Burst and gaps",
}
# The model's pattern names, shown to staff in plain words.
_PATTERN_LABELS: dict[str, str] = {
    "rose": "Rose", "fell": "Fell", "started": "Started after no activity", "stopped": "Stopped",
    "money_in_only": "Money in only", "money_out_only": "Money out only",
    "in_out_mix_changed": "Money in/out mix changed", "amounts_changed": "Amounts changed",
    "burst_peak": "Burst peak", "burst_rising": "Burst rising",
    "gap_changed": "Gap between transactions changed", "long_gap_before": "Long gap before transaction",
}
# CSV fields attached as evidence for the months the model selected.
_EVIDENCE_FIELDS: dict[ReviewCheckName, tuple[str, ...]] = {
    "activity_and_amount_change": ("txn_count_monthly", "total_amount"),
    "money_in_and_out": ("debit_count_monthly", "credit_count_monthly", "monthly_debit", "monthly_credit"),
    "burst_and_gaps": ("txn_count_monthly", "pct_burst", "pct_trx_gap"),
}
# CSV fields whose values an insight may quote, for the months it is about.
_INSIGHT_FIELDS: dict[ReviewCheckName, tuple[str, ...]] = {
    "activity_and_amount_change": ("txn_count_monthly", "total_amount", "max_amount"),
    "money_in_and_out": ("debit_count_monthly", "credit_count_monthly", "monthly_debit", "monthly_credit", "total_amount"),
}

_FORBIDDEN_TEXT = (
    "n/a", "insufficient", "nothing happened", "please provide",
    "launder", "illicit", "illegal", "fraud", "criminal", "terroris", "smurf",
)
_FORBIDDEN_PROFILE_TEXT = (
    "n/a", "please provide", "named p",
)
_MONTH_TOKEN = re.compile(r"^M?(\d{6})$")
_NO_MONTHS = "none"
_SAYS_NO_BURST = re.compile(r"\b(?:no|without|zero|absence of)\s+(?:\w+\s+){0,2}bursts?\b")
_SAYS_EVEN_SPACING = re.compile(r"\b(?:even(?:ly)?|regular(?:ly)?|consistent(?:ly)?)\s+(?:\w+\s+){0,1}(?:spaced|spread|spacing|timing|intervals?)\b")
_OUTCOME_SYNONYMS = {
    "pattern_found": "pattern_found", "found": "pattern_found", "observed": "pattern_found",
    "no_pattern_found": "no_pattern_found", "not_found": "no_pattern_found",
    "not_observed": "no_pattern_found", "no_pattern": "no_pattern_found",
}

_FALLBACK_HEADLINE = "The overall summary could not be produced. Review each check below."
_FALLBACK_RISK_REASON = "No LLM risk decision was available, so the account is treated as medium risk until reviewed."


class ModelOutputError(ValueError):
    """A model response was not complete, grounded, and safe to display."""


@dataclass(frozen=True)
class _Check:
    outcome: Outcome
    insight: str
    pattern: str
    months_text: str


@dataclass(frozen=True)
class _LlmCall:
    """One focused LLM call. Calls never share a prompt, so changing one cannot alter another."""

    stage: str
    checks: tuple[ReviewCheckName, ...]
    system_prompt: str
    prompt_input: str
    model_type: type[BaseModel]
    validator: Callable[[Any], Any]


class AnalysisService:
    """Four focused LLM calls with strict validation.

    Calls 1-3 run concurrently through the shared queue: change in activity and amounts
    with money in and out, activity after inactivity with burst and gaps, and the customer
    profile. Call 4 then reads their checked results with the CSV rows, writes the overall
    summary and decides the risk level. The model interprets; the application prepares
    labelled facts, checks the answers against the CSV, and lays out exact evidence.
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
        calls = self._pattern_calls(request.case_id, monthly, catalog, run)

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

        checks = self._build_checks(calls, results, monthly, catalog, run)
        if not results:
            return self._assessment(request, checks, None, profile_context)
        summary = await self._overall_summary_or_none(request.case_id, monthly, run, checks, profile, profile_context)
        return self._assessment(request, checks, summary, profile_context)

    def _pattern_calls(
        self, case_id: str, monthly: list[dict[str, Any]], catalog: dict[str, EvidenceItem],
        run: InactivityRun | None,
    ) -> list[_LlmCall]:
        return [
            _LlmCall(
                stage="activity-money-context",
                checks=("activity_and_amount_change", "money_in_and_out"),
                system_prompt=ACTIVITY_MONEY_SYSTEM_PROMPT,
                prompt_input=activity_money_input(monthly),
                model_type=_RawActivityMoney,
                validator=lambda raw: self._validate_activity_money(raw, catalog, monthly, case_id),
            ),
            _LlmCall(
                stage="inactivity-burst-gaps-context",
                checks=("activity_after_inactivity", "burst_and_gaps"),
                system_prompt=INACTIVITY_BURST_GAPS_SYSTEM_PROMPT,
                prompt_input=inactivity_burst_gaps_input(monthly, run, self._settings.dormancy_min_zero_months),
                model_type=_RawInactivityBurstGaps,
                validator=lambda raw: self._validate_inactivity_burst_gaps(raw, catalog, monthly, run, case_id),
            ),
        ]

    # --- validation: calls 1 and 2 -------------------------------------------------
    # Wrong months cost one retry (as before). A sentence or pattern name that does not
    # match the CSV never costs a retry: only that sentence is hidden and logged, while
    # the outcome, months and evidence table are kept.

    def _validate_activity_money(
        self, raw: "_RawActivityMoney", catalog: dict[str, EvidenceItem], monthly: list[dict[str, Any]], case_id: str,
    ) -> "_RawActivityMoney":
        known_months = self._known_months(catalog)
        by_month = {row["year_month"]: row for row in monthly}
        updates: dict[str, Any] = {}
        for name, prefix in (("activity_and_amount_change", "activity"), ("money_in_and_out", "money_flow")):
            check = raw.checks()[name]
            if check.outcome == "no_pattern_found":
                updates[f"{prefix}_months"], updates[f"{prefix}_pattern"] = _NO_MONTHS, "none"
                months: list[str] = []
            else:
                months = self._split_months(check.months_text, f"{name} months", min_count=2, max_count=2)
                self._validate_transaction_months(months, known_months, name)
                if months != sorted(months):
                    raise ModelOutputError(f"{name} months must be chronological")
                first, second = (by_month[month] for month in months)
                fields = _EVIDENCE_FIELDS[name]
                unchanged = all(Decimal(str(first[field])) == Decimal(str(second[field])) for field in fields)
                persistent_one_sided = name == "money_in_and_out" and (
                    (first["debit_count_monthly"] > 0 and second["debit_count_monthly"] > 0
                     and first["credit_count_monthly"] == second["credit_count_monthly"] == 0)
                    or (first["credit_count_monthly"] > 0 and second["credit_count_monthly"] > 0
                        and first["debit_count_monthly"] == second["debit_count_monthly"] == 0)
                )
                if unchanged and not persistent_one_sided:
                    raise ModelOutputError(f"{name} selected months show no change in the cited fields")
                updates[f"{prefix}_months"] = ",".join(months)
                problem = pattern_problem(name, check.pattern, monthly, months)
                if problem:
                    logger.warning("Insight hidden: case=%s check=%s reason=%s", case_id, name, problem)
                    updates[f"{prefix}_pattern"], updates[f"{prefix}_insight"] = "none", ""
                    continue
            allowed = quotable_numbers(monthly, months, _INSIGHT_FIELDS[name])
            problem = self._text_problem(check.insight, allowed)
            if problem:
                logger.warning("Insight hidden: case=%s check=%s reason=%s", case_id, name, problem)
                updates[f"{prefix}_insight"] = ""
        return raw.model_copy(update=updates)

    def _validate_inactivity_burst_gaps(
        self, raw: "_RawInactivityBurstGaps", catalog: dict[str, EvidenceItem], monthly: list[dict[str, Any]],
        run: InactivityRun | None, case_id: str,
    ) -> "_RawInactivityBurstGaps":
        updates: dict[str, Any] = {}
        # Activity after inactivity is a computed fact; the model only explains it.
        if run is None:
            updates["inactivity_insight"] = ""
        else:
            problem = self._text_problem(raw.inactivity_insight, self._inactivity_numbers(run, monthly))
            if problem:
                logger.warning("Insight hidden: case=%s check=activity_after_inactivity reason=%s", case_id, problem)
                updates["inactivity_insight"] = ""

        months: list[str] = []
        if raw.burst_gaps_outcome == "pattern_found":
            months = self._split_months(raw.burst_gaps_months, "burst_and_gaps months", min_count=1, max_count=2)
            self._validate_transaction_months(months, self._known_months(catalog), "burst_and_gaps")
            if months != sorted(months):
                raise ModelOutputError("burst_and_gaps months must be chronological")
            updates["burst_gaps_months"] = ",".join(months)
            problem = pattern_problem("burst_and_gaps", raw.burst_gaps_pattern, monthly, months)
            if problem:
                logger.warning("Insight hidden: case=%s check=burst_and_gaps reason=%s", case_id, problem)
                updates["burst_gaps_pattern"], updates["burst_gaps_insight"] = "none", ""
                return raw.model_copy(update=updates)
        else:
            updates["burst_gaps_months"], updates["burst_gaps_pattern"] = _NO_MONTHS, "none"
        problem = self._text_problem(raw.burst_gaps_insight, allowed_burst_gap_numbers(monthly))
        lowered = raw.burst_gaps_insight.casefold()
        if not problem and has_burst(monthly) and _SAYS_NO_BURST.search(lowered):
            problem = "says no burst, but a month has burst share above 0%"
        if not problem and not has_in_month_gap(monthly) and _SAYS_EVEN_SPACING.search(lowered):
            problem = "describes spacing, but no month has 2 or more transactions"
        if problem:
            logger.warning("Insight hidden: case=%s check=burst_and_gaps reason=%s", case_id, problem)
            updates["burst_gaps_insight"] = ""
        return raw.model_copy(update=updates)

    def _inactivity_numbers(self, run: InactivityRun, monthly: list[dict[str, Any]]) -> set[Decimal]:
        fields = ("txn_count_monthly", "debit_count_monthly", "credit_count_monthly", "total_amount",
                  "max_amount", "monthly_debit", "monthly_credit", "pct_trx_gap")
        allowed = quotable_numbers(monthly, [run.active_month], fields)
        extra = {Decimal(run.zero_months), Decimal(run.later_transactions), run.later_total,
                 Decimal(str(self._settings.dormancy_review_single_amount)),
                 Decimal(str(self._settings.dormancy_review_month_total))}
        return allowed | {value.normalize() for value in extra} | numbers_in_text(" ".join(map(str, extra)))

    @staticmethod
    def _text_problem(text: str, allowed: set[Decimal]) -> str | None:
        """Why a model sentence cannot be shown, or None when it is grounded and neutral."""
        if not text.strip():
            return None
        lowered = text.casefold()
        if any(phrase in lowered for phrase in _FORBIDDEN_TEXT):
            return "prohibited wording"
        token = unquotable_number(text, allowed)
        return f"number not in the figures for these months: {token}" if token else None

    # --- building the checks ---------------------------------------------------------

    def _build_checks(
        self, calls: list[_LlmCall], results: dict[str, Any], monthly: list[dict[str, Any]],
        catalog: dict[str, EvidenceItem], run: InactivityRun | None,
    ) -> list[ReviewCheck]:
        stage_of = {name: call.stage for call in calls for name in call.checks}
        return [self._build_check(name, results.get(stage_of[name]), monthly, catalog, run) for name in _CHECK_ORDER]

    def _build_check(
        self, name: ReviewCheckName, raw: Any, monthly: list[dict[str, Any]],
        catalog: dict[str, EvidenceItem], run: InactivityRun | None,
    ) -> ReviewCheck:
        if raw is None:
            return ReviewCheck(
                check=name, title=_CHECK_TITLES[name], outcome="not_verified",
                facts="The LLM result for this check could not be verified. Review the figures below directly.",
                table=self._table(name, monthly, []),
            )
        if name == "activity_after_inactivity":
            if run is None:
                return ReviewCheck(
                    check=name, title=_CHECK_TITLES[name], outcome="no_pattern_found",
                    facts=no_inactivity_rationale(self._settings.dormancy_min_zero_months),
                    table=self._table(name, monthly, []),
                )
            run_months = [row["year_month"] for row in monthly if run.zero_start <= row["year_month"] <= run.active_month]
            ids = [
                f"M{run.zero_start}.txn_count_monthly", f"M{run.zero_end}.txn_count_monthly",
                f"M{run.active_month}.txn_count_monthly", f"M{run.active_month}.total_amount",
                f"M{run.active_month}.max_amount", f"M{run.active_month}.monthly_debit",
                f"M{run.active_month}.monthly_credit",
            ]
            return ReviewCheck(
                check=name, title=_CHECK_TITLES[name], outcome="pattern_found",
                insight=raw.inactivity_insight.strip() or None, months=[run.zero_start, run.active_month],
                facts=dormancy_rationale(run), table=self._table(name, monthly, run_months),
                evidence=self._resolve(list(dict.fromkeys(ids)), catalog, name),
            )
        if name == "burst_and_gaps":
            months = [] if raw.burst_gaps_outcome != "pattern_found" else raw.burst_gaps_months.split(",")
            facts = burst_gap_pair_facts(monthly, months) if months else burst_gap_six_month_context(monthly)
            evidence_months = months or burst_gap_evidence_months(monthly)
            return ReviewCheck(
                check=name, title=_CHECK_TITLES[name], outcome=raw.burst_gaps_outcome,
                pattern=_PATTERN_LABELS.get(raw.burst_gaps_pattern), months=months,
                insight=raw.burst_gaps_insight.strip() or None, facts=facts,
                table=self._table(name, monthly, months),
                evidence=self._evidence_for_months(name, evidence_months, catalog),
            )
        check = raw.checks()[name]
        months = [] if check.outcome != "pattern_found" else check.months_text.split(",")
        if name == "activity_and_amount_change":
            facts = activity_pair_context(monthly, months) if months else activity_six_month_context(monthly)
        else:
            facts = flow_pair_context(monthly, months) if months else flow_six_month_context(monthly)
        return ReviewCheck(
            check=name, title=_CHECK_TITLES[name], outcome=check.outcome,
            pattern=_PATTERN_LABELS.get(check.pattern), months=months,
            insight=check.insight.strip() or None, facts=facts,
            table=self._table(name, monthly, months),
            evidence=self._evidence_for_months(name, months, catalog),
        )

    @staticmethod
    def _table(name: ReviewCheckName, monthly: list[dict[str, Any]], highlight: list[str]) -> EvidenceTable:
        columns, rows, highlighted = check_table(name, monthly, highlight)
        return EvidenceTable(columns=columns, rows=rows, highlighted_rows=highlighted)

    def _evidence_for_months(
        self, name: ReviewCheckName, months: list[str], catalog: dict[str, EvidenceItem],
    ) -> list[EvidenceItem]:
        ids = [f"M{month}.{field}" for month in months for field in _EVIDENCE_FIELDS[name]]
        return self._resolve(ids, catalog, name)

    # --- call 4: overall summary and risk --------------------------------------------

    async def _overall_summary_or_none(
        self, case_id: str, monthly: list[dict[str, Any]], run: InactivityRun | None, checks: list[ReviewCheck],
        profile: list[dict[str, str | None]], profile_context: CustomerProfileContext | None,
    ) -> "tuple[OverallSummary, RiskLevel] | None":
        prompt_input = overall_summary_input(
            monthly, run, self._settings.dormancy_min_zero_months,
            [self._check_line(check) for check in checks],
            [check.check for check in checks if check.outcome == "not_verified"],
            self._profile_lines(profile, profile_context),
        )
        quotable = numbers_in_text(prompt_input)
        try:
            raw = await self._complete_with_retry(
                case_id, "overall-summary", OVERALL_SUMMARY_SYSTEM_PROMPT, prompt_input, _RawOverallSummary,
                self._settings.llm_summary_max_response_tokens,
                lambda value: self._validate_overall_summary(value, quotable, case_id),
            )
        except Exception as exc:  # the report must still be shown without the summary
            logger.error("Overall summary failed: case=%s error=%s", case_id, exc)
            return None
        points = list(dict.fromkeys(point for point in (raw.point_1, raw.point_2, raw.point_3) if point))
        summary = OverallSummary(
            verified=True, headline=raw.headline, points=points, why_it_matters=raw.why_it_matters or None,
            verify_first=[item for item in (raw.verify_1, raw.verify_2) if item], risk_reason=raw.risk_reason,
        )
        return summary, raw.risk_level

    def _validate_overall_summary(
        self, raw: "_RawOverallSummary", quotable: set[Decimal], case_id: str,
    ) -> "_RawOverallSummary":
        """Drop any line that quotes a figure not in the input or uses alleging wording.

        A dropped headline is replaced by the first remaining point. Only a summary with
        nothing usable left (or no risk reason / verify step) costs the one retry.
        """
        updates: dict[str, Any] = {}
        for field in _RawOverallSummary.TEXT_FIELDS:
            problem = self._text_problem(getattr(raw, field), quotable)
            if problem:
                logger.warning("Summary line dropped: case=%s field=%s reason=%s", case_id, field, problem)
                updates[field] = ""
        checked = raw.model_copy(update=updates)
        points = [point for point in (checked.point_1, checked.point_2, checked.point_3) if point]
        if not checked.headline and points:
            checked = checked.model_copy(update={"headline": points[0]})
            points = points[1:]
        checked = checked.model_copy(update=dict(zip(("point_1", "point_2", "point_3"), points + ["", "", ""])))
        missing = [field for field in ("headline", "verify_1", "risk_reason") if not getattr(checked, field)]
        if not checked.verify_1 and checked.verify_2:
            checked, missing = checked.model_copy(update={"verify_1": checked.verify_2, "verify_2": ""}), \
                [field for field in missing if field != "verify_1"]
        if missing:
            raise ModelOutputError(f"overall summary has no usable {', '.join(missing)}")
        return checked

    @staticmethod
    def _check_line(check: ReviewCheck) -> str:
        months = ",".join(check.months) or "none"
        return (
            f"CHECK|{check.check}|{check.outcome}|pattern={check.pattern or 'none'}|months={months}|"
            f"insight={check.insight or 'none'}|facts={check.facts}"
        )

    @staticmethod
    def _profile_lines(
        profile: list[dict[str, str | None]], profile_context: CustomerProfileContext | None,
    ) -> list[str]:
        if not profile:
            return []
        latest = profile[-1]
        lines = [
            f"PROFILE|occupation={latest.get('occupation') or 'unknown'}|citizenship={latest.get('citizenship') or 'unknown'}|"
            f"individual_or_organisation={latest.get('indv_org_type') or 'unknown'}|"
            f"effective_from={latest.get('valid_from_dttm') or 'unknown'}|versions={len(profile)}"
        ]
        if profile_context is not None:
            lines.append(f"PROFILE_SUMMARY|{profile_context.summary}")
        return lines

    def _assessment(
        self, request: AccountAnalysisRequest, checks: list[ReviewCheck],
        summary: "tuple[OverallSummary, RiskLevel] | None", profile_context: CustomerProfileContext | None,
    ) -> AccountAssessment:
        all_checks_verified = all(check.outcome != "not_verified" for check in checks)
        if summary is None:
            overall, risk_level = OverallSummary(
                verified=False, headline=_FALLBACK_HEADLINE, risk_reason=_FALLBACK_RISK_REASON,
            ), "medium"
        else:
            overall, risk_level = summary
        complete = overall.verified and all_checks_verified
        decision: Decision = (
            "no_further_review_suggested" if complete and risk_level == "low" else "further_review_suggested"
        )
        limitations = self._static_limitations()
        unverified = [check.title for check in checks if check.outcome == "not_verified"]
        if unverified:
            limitations.append(AssessmentLimitation(limitation=(
                "Not verified by the LLM: " + ", ".join(unverified)
                + ". An authorised reviewer must assess these checks from the figures shown."
            )))
        if not overall.verified:
            limitations.append(AssessmentLimitation(
                limitation="The overall summary could not be produced, so no LLM risk decision is available.",
            ))
        return AccountAssessment(
            case_id=request.case_id, acct_num=request.acct_num,
            status="completed" if complete else "needs_review", decision=decision, risk_level=risk_level,
            overall=overall, review_checks=checks, customer_profile_context=profile_context,
            limitations=limitations, months_reviewed=len(request.monthly_summary),
            profile_records_matched=len(request.customer_profile), generated_at=datetime.now(timezone.utc),
        )

    # --- profile -------------------------------------------------------------------------

    def _hydrate_profile_context(
        self, raw: "_RawProfileContext", catalog: dict[str, EvidenceItem],
        profile: list[dict[str, str | None]], monthly: list[dict[str, Any]],
    ) -> CustomerProfileContext:
        latest = profile[-1]
        peak = max(monthly, key=lambda row: Decimal(str(row["total_amount"])))
        peak_single = max(monthly, key=lambda row: Decimal(str(row["max_amount"])))
        party_type = {"I": "Individual", "O": "Organisation"}.get(latest.get("indv_org_type") or "", latest.get("indv_org_type") or "-")
        rows = [
            ["Occupation", latest.get("occupation") or "-"],
            ["Citizenship", latest.get("citizenship") or "-"],
            ["Individual / organisation", party_type],
            ["Profile effective from", (latest.get("valid_from_dttm") or "-")[:10]],
            ["Profile last maintained", (latest.get("last_maint_dt") or "-")[:10]],
            ["Profile versions supplied", str(len(profile))],
            ["Highest monthly total", f"{peak['year_month']}: {Decimal(str(peak['total_amount'])):,.2f}"],
            ["Largest single amount", f"{peak_single['year_month']}: {Decimal(str(peak_single['max_amount'])):,.2f}"],
        ]
        return CustomerProfileContext(
            summary=raw.profile_summary,
            table=EvidenceTable(columns=["Field", "Value"], rows=rows),
            evidence=self._canonical_profile_evidence(catalog, profile, monthly),
        )

    # --- LLM plumbing -------------------------------------------------------------------

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
        prepare = getattr(model_type, "prepare_payload", None)
        if prepare is not None:
            result = prepare(result)
        try:
            return model_type.model_validate(result)
        except ValidationError as exc:
            raise ModelOutputError(f"{stage} output did not meet the required schema: {exc}") from exc

    # --- unchanged helpers (LLM retry, profile, evidence) ----------------------------

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

    def _inactivity_run(self, monthly: list[dict[str, Any]]) -> InactivityRun | None:
        settings = self._settings
        return find_inactivity_run(
            monthly, min_zero_months=settings.dormancy_min_zero_months,
            single_reference=Decimal(str(settings.dormancy_review_single_amount)),
            month_total_reference=Decimal(str(settings.dormancy_review_month_total)),
        )

    def _validate_profile_context(self, raw: "_RawProfileContext") -> None:
        self._reject_forbidden_text(raw.profile_summary, _FORBIDDEN_PROFILE_TEXT, "profile summary")

    @staticmethod
    def _reject_forbidden_text(value: str, phrases: tuple[str, ...], label: str) -> None:
        if any(phrase in value.casefold() for phrase in phrases):
            raise ModelOutputError(f"{label} contains a prohibited missing-data phrase")

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

    @staticmethod
    def _known_months(catalog: dict[str, EvidenceItem]) -> set[str]:
        return {item_id[1:7] for item_id in catalog if item_id.startswith("M")}

    # --- rendering ------------------------------------------------------------

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


def _cap(text: str, limit: int) -> str:
    """Shorten text to ``limit`` characters at a word boundary."""
    if len(text) <= limit:
        return text
    cut = text[:limit].rsplit(" ", 1)[0].rstrip(",;: ")
    return cut + "..."


class _RawAnswer(BaseModel):
    """Shared repairs for the flat JSON answer of one call. Repairs never change a decision."""

    model_config = ConfigDict(extra="forbid")
    _OUTCOME_KEYS: ClassVar[tuple[str, ...]] = ()
    _PATTERN_KEYS: ClassVar[tuple[str, ...]] = ()
    _TEXT_LIMITS: ClassVar[dict[str, int]] = {}

    @classmethod
    def prepare_payload(cls, payload: Any) -> Any:
        """Repair harmless formatting so it does not cost a retry.

        Unexpected keys are dropped (logged); outcome spelling such as "Pattern found" or
        "observed" is normalised; pattern names are lower-cased; an empty *_months becomes
        "none"; "none"/empty text becomes ""; text over its limit is shortened at a word.
        """
        if not isinstance(payload, dict):
            return payload
        extra = sorted(set(payload) - set(cls.model_fields))
        if extra:
            logger.warning("Ignoring unexpected key(s) from the model: %s", ", ".join(extra))
        payload = {key: value for key, value in payload.items() if key in cls.model_fields}
        for key in cls._OUTCOME_KEYS:
            if isinstance(payload.get(key), str):
                spelled = re.sub(r"[\s-]+", "_", payload[key].strip().lower())
                payload[key] = _OUTCOME_SYNONYMS.get(spelled, spelled)
        for key in cls._PATTERN_KEYS:
            if isinstance(payload.get(key), str):
                payload[key] = re.sub(r"[\s-]+", "_", payload[key].strip().lower()) or "none"
        for key in cls.model_fields:
            if key.endswith("_months") and isinstance(payload.get(key), str) and not payload[key].strip():
                payload[key] = _NO_MONTHS
        for key, limit in cls._TEXT_LIMITS.items():
            value = payload.get(key)
            if isinstance(value, str):
                value = "" if value.strip().casefold() in {"none", "n/a", "null", "-"} else value.strip()
                payload[key] = _cap(value, limit)
        return payload


class _RawActivityMoney(_RawAnswer):
    _OUTCOME_KEYS: ClassVar[tuple[str, ...]] = ("activity_outcome", "money_flow_outcome")
    _PATTERN_KEYS: ClassVar[tuple[str, ...]] = ("activity_pattern", "money_flow_pattern")
    _TEXT_LIMITS: ClassVar[dict[str, int]] = {"activity_insight": 300, "money_flow_insight": 300}
    activity_insight: str = Field(default="", max_length=300)
    activity_pattern: str = Field(default="none", max_length=40)
    activity_months: str = Field(min_length=1, max_length=32)
    activity_outcome: Outcome
    money_flow_insight: str = Field(default="", max_length=300)
    money_flow_pattern: str = Field(default="none", max_length=40)
    money_flow_months: str = Field(min_length=1, max_length=32)
    money_flow_outcome: Outcome

    def checks(self) -> dict[ReviewCheckName, _Check]:
        return {
            "activity_and_amount_change": _Check(
                self.activity_outcome, self.activity_insight, self.activity_pattern, self.activity_months),
            "money_in_and_out": _Check(
                self.money_flow_outcome, self.money_flow_insight, self.money_flow_pattern, self.money_flow_months),
        }


class _RawInactivityBurstGaps(_RawAnswer):
    _OUTCOME_KEYS: ClassVar[tuple[str, ...]] = ("burst_gaps_outcome",)
    _PATTERN_KEYS: ClassVar[tuple[str, ...]] = ("burst_gaps_pattern",)
    _TEXT_LIMITS: ClassVar[dict[str, int]] = {"inactivity_insight": 300, "burst_gaps_insight": 300}
    inactivity_insight: str = Field(default="", max_length=300)
    burst_gaps_insight: str = Field(default="", max_length=300)
    burst_gaps_pattern: str = Field(default="none", max_length=40)
    burst_gaps_months: str = Field(min_length=1, max_length=32)
    burst_gaps_outcome: Outcome


class _RawOverallSummary(_RawAnswer):
    TEXT_FIELDS: ClassVar[tuple[str, ...]] = (
        "headline", "point_1", "point_2", "point_3", "why_it_matters", "verify_1", "verify_2", "risk_reason",
    )
    _TEXT_LIMITS: ClassVar[dict[str, int]] = {
        "headline": 160, "point_1": 300, "point_2": 300, "point_3": 300, "why_it_matters": 360,
        "verify_1": 260, "verify_2": 260, "risk_reason": 260,
    }
    headline: str = Field(default="", max_length=160)
    point_1: str = Field(default="", max_length=300)
    point_2: str = Field(default="", max_length=300)
    point_3: str = Field(default="", max_length=300)
    why_it_matters: str = Field(default="", max_length=360)
    verify_1: str = Field(default="", max_length=260)
    verify_2: str = Field(default="", max_length=260)
    risk_reason: str = Field(default="", max_length=260)
    risk_level: RiskLevel

    @classmethod
    def prepare_payload(cls, payload: Any) -> Any:
        payload = super().prepare_payload(payload)
        if isinstance(payload, dict) and isinstance(payload.get("risk_level"), str):
            payload["risk_level"] = payload["risk_level"].strip().lower()
        return payload


class _RawProfileContext(BaseModel):
    model_config = ConfigDict(extra="forbid")
    profile_summary: str = Field(min_length=1, max_length=650)
