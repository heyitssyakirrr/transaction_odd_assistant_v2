from __future__ import annotations

import asyncio
import logging
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Callable, ClassVar, Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, ValidationError

from app.adapters.llm_client import LlmOutputFormatError, LlmOutputTruncatedError
from app.config import Settings
from app.core.call_timing import CallMeasure
from app.core.llm_work_queue import LlmWorkQueue
from app.core.monthly_facts import (
    PATTERN_LABELS, InactivityRun, Screening, activity_all_months_context, activity_months_context,
    allowed_burst_gap_numbers, burst_gap_evidence_months, burst_gap_all_months_context, burst_gap_pair_facts,
    calendar_numbers, canonical_pattern, check_table, dormancy_rationale, find_inactivity_run, burst_guide_months,
    has_in_month_gap, layering_context, months_label, no_inactivity_rationale, numbers_in_text, pattern_problem, quotable_numbers,
    RiskSignals, layering_months, risk_because, risk_signals, screening_risk, unmatched_month_name, unquotable_number,
    unusual_transaction_months,
)
from app.core.models import (
    AccountAnalysisRequest, AccountAssessment, AnalysisTiming, AssessmentLimitation, CallTiming,
    CustomerProfileContext, CustomerProfileRecord, Decision, EvidenceItem, EvidenceTable, LlmClient, OverallSummary,
    ReviewCheck, ReviewCheckName, RiskLevel,
)
from app.core.prompts import (
    INACTIVITY_BURST_GAPS_SYSTEM_PROMPT,
    PROFILE_CONTEXT_SYSTEM_PROMPT, account_notes, activity_money_input, activity_money_prompt, format_retry_suffix,
    inactivity_burst_gaps_input, overall_summary_input, overall_summary_prompt, profile_context_input,
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
    "money_in_and_out": "Layering (money in and money out)",
    "burst_and_gaps": "Burst and gaps",
}
# CSV fields attached as evidence for the months selected (at most 3 per month: 36 items for 12 months).
_EVIDENCE_FIELDS: dict[ReviewCheckName, tuple[str, ...]] = {
    "activity_and_amount_change": ("txn_count_monthly", "total_amount", "max_amount"),
    "money_in_and_out": ("monthly_credit", "monthly_debit"),
    "burst_and_gaps": ("txn_count_monthly", "pct_burst", "pct_trx_gap"),
}
# The two checks the code screens (call 1 only describes them), with the pattern each one finds.
_SCREENED_PATTERNS: dict[ReviewCheckName, str] = {
    "activity_and_amount_change": "large_transaction", "money_in_and_out": "layering",
}
def _sentence_issue(text: str) -> str:
    """Why an explanation does not count as one, in plain words for staff."""
    return "Is a label, not a sentence: the AI wrote a pattern name or a few words instead of an explanation."
_SAYS_NO_INACTIVITY = re.compile(r"\bno (?:inactive|quiet|dormant) period\b", re.IGNORECASE)
# "No activity in the six/12 months" when transactions exist (seen for one-transaction accounts).
# Words the summary is told not to use when nothing was found (the risk is low); shown as an issue if used.
_SUSPICION_WORDS = re.compile(r"money laundering|suspicious|suspicion|red flag", re.IGNORECASE)
_FIT_VERDICTS = ("Fits the occupation:", "Does not fit the occupation:")
_SAYS_NO_ACTIVITY = re.compile(
    r"\bno (?:activity|transactions?|debit or credit flow|flow)\b[^.]{0,30}\b(?:six|twelve|\d+)[- ]months?\b",
    re.IGNORECASE)
_SUMMARY_FIELD_NAMES = {
    "headline": "Headline", "point_1": "Point 1", "point_2": "Point 2", "point_3": "Point 3",
    "why_it_matters": "Why it matters", "occupation_fit": "Fit with occupation", "verify_1": "Verify 1", "verify_2": "Verify 2", "risk_reason": "Risk reason",
}
# Summary lines kept even when they are not a sentence: the summary cannot be shown without them.
_SUMMARY_REQUIRED_FIELDS = ("headline", "risk_reason")
# ReviewCheck.facts allows 400 characters; facts built from many months are cut at a word.
_FACTS_LIMIT = 400

# Phrases that mean the model gave no real answer. Judgement words are allowed: the
# output supports the reviewer, who makes the decision.
_NON_ANSWER_TEXT = ("n/a", "insufficient data", "insufficient information", "nothing happened", "please provide")
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

# The four LLM calls, in the order staff read them on the timing table.
_TIMED_CALLS: tuple[tuple[str, str], ...] = (
    ("activity-money-context", "Call 1: Activity and money"),
    ("inactivity-burst-gaps-context", "Call 2: Inactivity, burst and gaps"),
    ("profile-context", "Call 3: Customer profile"),
    ("overall-summary", "Call 4: Overall summary"),
)
# occupation_fit when there is nothing for the AI to judge (it is not asked; see occupation_note).
_NOTHING_TO_COMPARE = "No unusual transaction or layering was found, so there is nothing to compare with the occupation."
_NO_OCCUPATION = "The declared occupation is not available, so the findings could not be compared with it."


class ModelOutputError(ValueError):
    """A model response was not complete, grounded, and safe to display."""


def _pattern_label(name: str) -> str | None:
    """The staff label for a pattern name; an unrecognised name is shown as the model wrote it."""
    return None if name == "none" else PATTERN_LABELS.get(name, name)


def _is_sentence(text: str) -> bool:
    """Three or more words. "No", "None given" or "debit_only" is a label, not an explanation."""
    return len(text.split()) >= 3


def _sentence_issues(key: str, text: str, retry_allowed: bool) -> list[str]:
    """Check one explanation: retry when it is missing or only a label.

    "debit_only" (under three words) is a label, not an explanation: it costs the one
    retry when ``retry_allowed``, and is shown with that issue after it. Any real
    sentence is accepted as written; its figures are checked separately.
    """
    if _is_sentence(text):
        return []
    if retry_allowed:
        raise _sentence_error(key)
    return [_sentence_issue(text)] if text else []


def _schema_problems(exc: ValidationError) -> str:
    """A validation error as a short instruction, e.g. "money_flow_outcome is missing"."""
    problems = []
    for error in exc.errors():
        key = ".".join(str(part) for part in error["loc"]) or "the answer"
        problems.append(f"{key} is missing" if error["type"] == "missing" else f"{key}: {error['msg']}")
    return "; ".join(problems) + ". Return every key in one JSON object"


def _sentence_error(key: str) -> ModelOutputError:
    return ModelOutputError(f"{key} must be one or two full sentences for staff, not a pattern name or none")


@dataclass(frozen=True)
class _LlmCall:
    """One focused LLM call. Calls never share a prompt, so changing one cannot alter another."""

    stage: str
    checks: tuple[ReviewCheckName, ...]
    system_prompt: str
    prompt_input: str
    model_type: type[BaseModel]
    # Called as validator(answer, final_attempt); may return a corrected copy.
    validator: Callable[[Any, bool], Any]


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
        started = time.monotonic()
        measures = {stage: CallMeasure() for stage, _ in _TIMED_CALLS}
        profile = self._profile_records_for_llm(request.customer_profile)
        catalog = self._evidence_catalog(request, profile)
        monthly = [row.model_dump(mode="json") for row in request.monthly_summary]
        run = self._inactivity_run(monthly)
        screening = self._screening()
        screened: dict[ReviewCheckName, list[str]] = {
            "activity_and_amount_change": unusual_transaction_months(monthly, screening),
            "money_in_and_out": layering_months(monthly, screening),
        }
        logger.info(
            "Screening: case=%s unusual_transaction=%s layering=%s", request.case_id,
            ",".join(screened["activity_and_amount_change"]) or "none", ",".join(screened["money_in_and_out"]) or "none",
        )
        calls = self._pattern_calls(request.case_id, monthly, catalog, run, screening)

        call_tasks = [
            asyncio.create_task(self._complete_with_retry(
                request.case_id, call.stage, call.system_prompt, call.prompt_input, call.model_type,
                self._settings.llm_transaction_max_response_tokens, call.validator, measures[call.stage],
            ))
            for call in calls
        ]
        profile_task = asyncio.create_task(
            self._profile_context_or_none(request.case_id, profile, monthly, measures["profile-context"]))
        parallel_started = time.monotonic()
        *call_results, profile_result = await asyncio.gather(*call_tasks, profile_task, return_exceptions=True)
        parallel_seconds = time.monotonic() - parallel_started

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

        checks = self._build_checks(calls, results, monthly, catalog, run, screened)
        signals = risk_signals(monthly, screening)
        summary = None if not results else await self._overall_summary_or_none(
            request.case_id, monthly, run, checks, profile, profile_context, signals, measures["overall-summary"])
        assessment = self._assessment(request, checks, summary, signals, profile_context)
        timing = self._timing(measures, started, parallel_seconds)
        logger.info(
            "Timing: case=%s %s calls_1_to_3=%.1fs total=%.1fs", request.case_id,
            " ".join(f"call{index}={call.llm_seconds if call.llm_seconds is not None else '-'}s"
                     f"/{call.output_tokens if call.output_tokens is not None else '-'}tok"
                     for index, call in enumerate(timing.calls, start=1)),
            timing.calls_1_to_3_seconds, timing.total_seconds,
        )
        return assessment.model_copy(update={"timing": timing})

    @staticmethod
    def _timing(measures: dict[str, CallMeasure], started: float, parallel_seconds: float) -> AnalysisTiming:
        """Each call's time from when a worker sent it to the LLM, and the account's total time."""
        calls = []
        for stage, label in _TIMED_CALLS:
            measure = measures[stage]
            if not measure.attempts:
                calls.append(CallTiming(call=label, status="not_run"))
                continue
            llm_seconds = measure.llm_ms / 1000
            calls.append(CallTiming(
                call=label, status="failed" if measure.failed else "answered",
                llm_seconds=round(llm_seconds, 1),
                started_after_seconds=round((measure.started_at or started) - started, 1),
                queue_wait_seconds=round(measure.queue_wait_ms / 1000, 1), attempts=measure.attempts,
                output_tokens=measure.output_tokens,
                tokens_per_second=(round(measure.output_tokens / llm_seconds, 1)
                                   if measure.output_tokens and llm_seconds > 0 else None),
            ))
        return AnalysisTiming(
            calls=calls, calls_1_to_3_seconds=round(parallel_seconds, 1),
            total_seconds=round(time.monotonic() - started, 1),
        )

    def _pattern_calls(
        self, case_id: str, monthly: list[dict[str, Any]], catalog: dict[str, EvidenceItem],
        run: InactivityRun | None, screening: Screening,
    ) -> list[_LlmCall]:
        activity_prompt, activity_input = activity_money_prompt(monthly, screening), activity_money_input(monthly)
        # The descriptions may quote any figure the AI was given (facts and notes), and only those.
        activity_numbers = numbers_in_text(activity_prompt + activity_input) | calendar_numbers(monthly)
        return [
            _LlmCall(
                stage="activity-money-context",
                checks=("activity_and_amount_change", "money_in_and_out"),
                system_prompt=activity_prompt,
                prompt_input=activity_input,
                model_type=_RawActivityMoney,
                validator=lambda raw, final: self._validate_activity_money(raw, monthly, activity_numbers, case_id, final),
            ),
            _LlmCall(
                stage="inactivity-burst-gaps-context",
                checks=("activity_after_inactivity", "burst_and_gaps"),
                system_prompt=INACTIVITY_BURST_GAPS_SYSTEM_PROMPT + account_notes(
                    monthly, run, self._settings.dormancy_min_zero_months, self._long_gap_days()),
                prompt_input=inactivity_burst_gaps_input(
                    monthly, run, self._settings.dormancy_min_zero_months, self._long_gap_days()),
                model_type=_RawInactivityBurstGaps,
                validator=lambda raw, final: self._validate_inactivity_burst_gaps(raw, catalog, monthly, run, case_id, final),
            ),
        ]

    # --- validation: calls 1 and 2 -------------------------------------------------
    # Each check in an answer is judged on its own. A check whose answer is incomplete
    # (a missing outcome, months that are not in the data, a found pattern without a
    # real sentence) costs one retry, and the retry is told exactly what was wrong. If the
    # retry is still incomplete, only that check is marked not verified; the other check
    # in the same answer is kept.
    # Nothing the model wrote is ever hidden: a sentence or pattern name that does not
    # match the CSV is shown as written, with the problem listed under it for staff.

    def _validate_activity_money(
        self, raw: "_RawActivityMoney", monthly: list[dict[str, Any]], allowed_numbers: set[Decimal],
        case_id: str, final_attempt: bool,
    ) -> "_RawActivityMoney":
        """Call 1 only describes the screening's findings. A missing or label-only description costs
        the one retry; anything else (a figure it was not given, a wrong month) is listed for staff."""
        # A description may name any month with transactions. A month with none cannot hold a
        # transaction, so naming it is still caught (logged 2026-10-05: "RM 10,000.00 in June
        # 2025" for a 202511 transaction, June 2025 being a quiet month).
        active_months = [row["year_month"] for row in monthly if row["txn_count_monthly"] > 0]
        issues: dict[str, list[str]] = {}
        for name, key in _RawActivityMoney.INSIGHT_KEYS.items():
            text = getattr(raw, key)
            found = _sentence_issues(key, text, retry_allowed=not final_attempt)
            found += self._text_issues(text, allowed_numbers, active_months, figures="the AI was given")
            if active_months and _SAYS_NO_ACTIVITY.search(text):
                found.append(f"Says there was no activity in the {months_label(monthly)}, but transactions were recorded.")
            issues[name] = self._log_issues(case_id, name, found)
        return self._with_issues(raw, issues)

    def _validate_inactivity_burst_gaps(
        self, raw: "_RawInactivityBurstGaps", catalog: dict[str, EvidenceItem], monthly: list[dict[str, Any]],
        run: InactivityRun | None, case_id: str, final_attempt: bool,
    ) -> "_RawInactivityBurstGaps":
        updates: dict[str, Any] = {}
        issues: dict[str, list[str]] = {}
        problems: dict[str, str] = {}
        # Activity after inactivity is a computed fact; the model only explains it.
        if run is None:
            updates["inactivity_insight"] = ""
        else:
            try:
                issues["activity_after_inactivity"] = self._inactivity_issues(
                    raw.inactivity_insight, run, monthly, case_id, final_attempt)
            except ModelOutputError as exc:
                problems["activity_after_inactivity"] = str(exc)
        try:
            updates.update(self._checked_burst_gaps(raw, catalog, monthly, case_id, final_attempt, issues))
        except ModelOutputError as exc:
            problems["burst_and_gaps"] = str(exc)
        return self._accepted(raw, updates, issues, problems, case_id, final_attempt, ("burst_and_gaps",))

    def _inactivity_issues(
        self, insight: str, run: InactivityRun, monthly: list[dict[str, Any]], case_id: str, final_attempt: bool,
    ) -> list[str]:
        """The inactive run is computed by the code; Qwen explains it. A missing explanation
        costs the one retry; anything else is listed for staff."""
        found = _sentence_issues("inactivity_insight", insight, retry_allowed=not final_attempt)
        if _SAYS_NO_INACTIVITY.search(insight):
            found.append(f"Says there was no inactive period, but the account had {run.zero_months} months with "
                         f"no transactions before {run.active_month}.")
        found += self._text_issues(insight, self._inactivity_numbers(run, monthly), self._run_months(run, monthly))
        return self._log_issues(case_id, "activity_after_inactivity", found)

    def _checked_burst_gaps(
        self, raw: "_RawInactivityBurstGaps", catalog: dict[str, EvidenceItem], monthly: list[dict[str, Any]],
        case_id: str, final_attempt: bool, issues: dict[str, list[str]],
    ) -> dict[str, Any]:
        if raw.burst_gaps_outcome is None:
            raise ModelOutputError("burst_gaps_outcome is missing; it must be pattern_found or no_pattern_found")
        # The sentence covers both bursts and gaps, so it always needs a month or figure.
        found = _sentence_issues("burst_gaps_insight", raw.burst_gaps_insight, retry_allowed=not final_attempt)
        updates: dict[str, Any] = {"burst_gaps_months": _NO_MONTHS, "burst_gaps_pattern": "none"}
        if raw.burst_gaps_outcome == "pattern_found":
            months = self._split_months(
                # Any of the months may be selected (Qwen often lists every quiet month).
                raw.burst_gaps_months or "", "burst_gaps_months", min_count=1, max_count=len(monthly))
            self._validate_transaction_months(months, self._known_months(catalog), "burst_and_gaps")
            pattern = canonical_pattern("burst_and_gaps", raw.burst_gaps_pattern)
            updates = {"burst_gaps_months": ",".join(months), "burst_gaps_pattern": pattern}
            if problem := pattern_problem("burst_and_gaps", pattern, monthly, months, long_gap_days=self._long_gap_days()):
                found.append(f"Pattern label: {problem}")
        all_months = [row["year_month"] for row in monthly]
        found += self._text_issues(raw.burst_gaps_insight, allowed_burst_gap_numbers(monthly), all_months)
        lowered = raw.burst_gaps_insight.casefold()
        if burst_guide_months(monthly) and _SAYS_NO_BURST.search(lowered):
            found.append("Says there was no burst, but a month has a burst share of 25% or more.")
        if not has_in_month_gap(monthly) and _SAYS_EVEN_SPACING.search(lowered):
            found.append("Describes the spacing of transactions, but no month has 2 or more transactions.")
        issues["burst_and_gaps"] = self._log_issues(case_id, "burst_and_gaps", found)
        return updates

    def _accepted(
        self, raw: Any, updates: dict[str, Any], issues: dict[str, list[str]], problems: dict[str, str],
        case_id: str, final_attempt: bool, decided: tuple[str, ...],
    ) -> Any:
        """Accept the answer, retry it once, or keep the complete checks and mark the rest not verified.

        ``decided`` names the checks this answer decides; when none of them is usable the
        whole call fails, exactly as an unreadable answer does.
        """
        if problems and (not final_attempt or set(decided) <= set(problems)):
            raise ModelOutputError("; ".join(problems.values()))
        for name, problem in problems.items():
            logger.warning("Check not verified: case=%s check=%s problem=%s", case_id, name, problem)
        result = raw.model_copy(update=updates)
        result.issues, result.unverified = issues, problems
        return result

    @staticmethod
    def _log_issues(case_id: str, check: str, issues: list[str]) -> list[str]:
        for issue in issues:
            logger.warning("Insight issue: case=%s check=%s issue=%s", case_id, check, issue)
        return issues

    @staticmethod
    def _with_issues(raw: Any, issues: dict[str, list[str]]) -> Any:
        raw.issues = issues
        return raw

    def _screening(self) -> Screening:
        """The compliance-owned screening settings (see app/config.py)."""
        settings = self._settings
        return Screening(
            unusual_times_usual=Decimal(str(settings.review_unusual_times_usual)),
            min_amount=Decimal(str(settings.review_min_amount)),
            layering_max_difference_pct=Decimal(str(settings.review_layering_max_difference_pct)),
            layering_times_usual_month=Decimal(str(settings.review_layering_times_usual_month)),
        )

    def _long_gap_days(self) -> Decimal:
        return Decimal(self._settings.review_long_gap_days)

    @staticmethod
    def _run_months(run: InactivityRun, monthly: list[dict[str, Any]]) -> list[str]:
        """The zero months of the inactive run and the month activity resumed."""
        return [row["year_month"] for row in monthly if run.zero_start <= row["year_month"] <= run.active_month]

    def _inactivity_numbers(self, run: InactivityRun, monthly: list[dict[str, Any]]) -> set[Decimal]:
        fields = ("txn_count_monthly", "debit_count_monthly", "credit_count_monthly", "total_amount",
                  "max_amount", "monthly_debit", "monthly_credit", "pct_trx_gap")
        allowed = quotable_numbers(monthly, [run.active_month], fields)
        extra = {Decimal(run.zero_months), Decimal(run.later_transactions), run.later_total}
        return allowed | {value.normalize() for value in extra} | numbers_in_text(" ".join(map(str, extra)))

    @staticmethod
    def _text_issues(
        text: str, allowed: set[Decimal], months: list[str], figures: str = "for the months this check is about",
    ) -> list[str]:
        """Where a model sentence does not fit the CSV, in plain words for staff (empty when it fits).

        ``figures`` says where a quoted number should have come from, for the issue text.
        """
        if not text.strip():
            return []
        issues = []
        if any(phrase in text.casefold() for phrase in _NON_ANSWER_TEXT):
            issues.append("Is not a real answer (for example N/A or a request for more data).")
        if token := unquotable_number(text, allowed):
            issues.append(f"Quotes {token}, which is not a figure {figures}.")
        if name := unmatched_month_name(text, months):
            issues.append(f"Mentions {name}, which is not one of the months this check is about.")
        return issues

    # --- building the checks ---------------------------------------------------------

    def _build_checks(
        self, calls: list[_LlmCall], results: dict[str, Any], monthly: list[dict[str, Any]],
        catalog: dict[str, EvidenceItem], run: InactivityRun | None, screened: dict[ReviewCheckName, list[str]],
    ) -> list[ReviewCheck]:
        stage_of = {name: call.stage for call in calls for name in call.checks}
        return [
            self._screened_check(name, results.get(stage_of[name]), monthly, catalog, screened[name])
            if name in screened else self._build_check(name, results.get(stage_of[name]), monthly, catalog, run)
            for name in _CHECK_ORDER
        ]

    def _screened_check(
        self, name: ReviewCheckName, raw: "_RawActivityMoney | None", monthly: list[dict[str, Any]],
        catalog: dict[str, EvidenceItem], months: list[str],
    ) -> ReviewCheck:
        """A check the code screened: its outcome and months are the screening's, its explanation
        the AI's. If call 1 failed, the finding still stands with its figures."""
        insight = getattr(raw, _RawActivityMoney.INSIGHT_KEYS[name], "").strip() if raw is not None else ""
        issues = raw.issues.get(name, []) if raw is not None else []
        if not insight:
            issues = [*issues, "No AI description was produced; the finding and figures below are from the CSV."]
        if name == "activity_and_amount_change":
            facts = activity_months_context(monthly, months) if months else activity_all_months_context(monthly)
        else:
            facts = layering_context(monthly, months)
        return ReviewCheck(
            check=name, title=_CHECK_TITLES[name], outcome="pattern_found" if months else "no_pattern_found",
            pattern=_pattern_label(_SCREENED_PATTERNS[name]) if months else None, months=months,
            insight=insight or None, issues=issues, facts=_cap(facts, _FACTS_LIMIT), table=self._table(name, monthly, months),
            evidence=self._evidence_for_months(name, months, catalog),
        )

    def _build_check(
        self, name: ReviewCheckName, raw: Any, monthly: list[dict[str, Any]],
        catalog: dict[str, EvidenceItem], run: InactivityRun | None,
    ) -> ReviewCheck:
        if raw is None or name in raw.unverified:
            return ReviewCheck(
                check=name, title=_CHECK_TITLES[name], outcome="not_verified",
                facts="The LLM result for this check could not be verified. Review the figures below directly.",
                issues=[f"Incomplete AI answer after one retry: {raw.unverified[name]}."] if raw is not None else [],
                table=self._table(name, monthly, []),
            )
        if name == "activity_after_inactivity":
            if run is None:
                return ReviewCheck(
                    check=name, title=_CHECK_TITLES[name], outcome="no_pattern_found",
                    facts=no_inactivity_rationale(self._settings.dormancy_min_zero_months, len(monthly)),
                    table=self._table(name, monthly, []),
                )
            run_months = self._run_months(run, monthly)
            ids = [
                f"M{run.zero_start}.txn_count_monthly", f"M{run.zero_end}.txn_count_monthly",
                f"M{run.active_month}.txn_count_monthly", f"M{run.active_month}.total_amount",
                f"M{run.active_month}.max_amount", f"M{run.active_month}.monthly_debit",
                f"M{run.active_month}.monthly_credit",
            ]
            return ReviewCheck(
                check=name, title=_CHECK_TITLES[name], outcome="pattern_found",
                insight=raw.inactivity_insight.strip() or None, issues=raw.issues.get(name, []),
                months=[run.zero_start, run.active_month],
                facts=dormancy_rationale(run), table=self._table(name, monthly, run_months),
                evidence=self._resolve(list(dict.fromkeys(ids)), catalog, name),
            )
        if name == "burst_and_gaps":
            months = [] if raw.burst_gaps_outcome != "pattern_found" else raw.burst_gaps_months.split(",")
            facts = burst_gap_pair_facts(monthly, months) if months else burst_gap_all_months_context(monthly)
            evidence_months = months or burst_gap_evidence_months(monthly)
            return ReviewCheck(
                check=name, title=_CHECK_TITLES[name], outcome=raw.burst_gaps_outcome,
                pattern=_pattern_label(raw.burst_gaps_pattern), months=months,
                insight=raw.burst_gaps_insight.strip() or None, issues=raw.issues.get(name, []),
                facts=_cap(facts, _FACTS_LIMIT),
                table=self._table(name, monthly, months),
                evidence=self._evidence_for_months(name, evidence_months, catalog),
            )
        raise ValueError(f"{name} is a screened check; build it with _screened_check")

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
        profile: list[dict[str, str | None]], profile_context: CustomerProfileContext | None, signals: RiskSignals,
        measure: CallMeasure | None = None,
    ) -> OverallSummary | None:
        """Call 4 explains the screening's findings and risk, and judges whether the evidence fits the
        occupation. The risk level itself is the screening's (see screening_risk)."""
        occupation = self._declared_occupation(profile)
        system_prompt = overall_summary_prompt(signals, occupation)
        prompt_input = overall_summary_input(
            monthly, run, self._settings.dormancy_min_zero_months,
            [self._check_line(check) for check in checks],
            [check.check for check in checks if check.outcome == "not_verified"],
            self._profile_lines(profile, profile_context), signals,
        )
        quotable = numbers_in_text(system_prompt + prompt_input) | calendar_numbers(monthly)
        try:
            raw = await self._complete_with_retry(
                case_id, "overall-summary", system_prompt, prompt_input,
                _RawOverallSummary, self._settings.llm_summary_max_response_tokens,
                lambda value, final: self._validate_overall_summary(
                    value, quotable, [row["year_month"] for row in monthly], case_id, final,
                    nothing_found=not (signals.unusual or signals.layering),
                    fit_expected=bool((signals.unusual or signals.layering) and occupation)),
                measure,
            )
        except Exception as exc:  # the report must still be shown without the summary
            logger.error("Overall summary failed: case=%s error=%s", case_id, exc)
            return None
        points = list(dict.fromkeys(point for point in (raw.point_1, raw.point_2, raw.point_3) if point))
        return OverallSummary(
            verified=True, headline=raw.headline, points=points, why_it_matters=raw.why_it_matters or None,
            occupation_fit=self._occupation_fit(raw.occupation_fit, signals, occupation, case_id),
            verify_first=[item for item in (raw.verify_1, raw.verify_2) if item], risk_reason=raw.risk_reason,
            issues=raw.issues.get("overall_summary", []),
        )

    @staticmethod
    def _declared_occupation(profile: list[dict[str, str | None]]) -> str | None:
        """The latest declared occupation and type, e.g. "Housewife, individual"; None when not available."""
        if not profile:
            return None
        latest = profile[-1]
        occupation = latest.get("occupation") or ""
        if not occupation or occupation.startswith("Unmapped occupation code"):
            return None
        party = {"I": "individual", "O": "organisation"}.get(latest.get("indv_org_type") or "")
        return f"{occupation}, {party}" if party else occupation

    @staticmethod
    def _occupation_fit(text: str, signals: RiskSignals, occupation: str | None, case_id: str) -> str | None:
        """The AI's fit judgement when there was evidence to judge; otherwise the reason there is none.
        The AI is told to write "none" then; anything it writes instead is logged, not shown."""
        if (signals.unusual or signals.layering) and occupation:
            return text or None
        if text:
            logger.warning("Occupation fit written with nothing to judge; not shown: case=%s text=%s", case_id, text)
        return _NOTHING_TO_COMPARE if not (signals.unusual or signals.layering) else _NO_OCCUPATION

    def _validate_overall_summary(
        self, raw: "_RawOverallSummary", quotable: set[Decimal], months: list[str], case_id: str, final_attempt: bool,
        *, nothing_found: bool = False, fit_expected: bool = False,
    ) -> "_RawOverallSummary":
        """Keep every line as written and list any that quote a figure or month not in the input.

        A line that is not a sentence (e.g. point_2 "No", logged 2026-10-05) is left out and
        listed as an issue; the headline and risk reason are kept with the issue, because
        the summary cannot be shown without them. A summary without a headline
        or risk reason costs the one retry. A missing verify step costs the retry only on
        the first attempt; on the final attempt it is kept without. Where the AI ignored a note
        (suspicion wording with nothing found, a fit without its verdict), the line is kept and
        the problem listed, never rewritten.
        """
        checked, not_sentences = self._without_non_sentences(raw)
        if not checked.verify_1 and checked.verify_2:
            checked = checked.model_copy(update={"verify_1": checked.verify_2, "verify_2": ""})
        required = _SUMMARY_REQUIRED_FIELDS if final_attempt else (*_SUMMARY_REQUIRED_FIELDS, "verify_1")
        missing = [field for field in required if not getattr(checked, field)]
        if missing:
            raise ModelOutputError(f"overall summary has no {', '.join(missing)}")
        issues = not_sentences + [
            f"{_SUMMARY_FIELD_NAMES[field]}: {issue}"
            for field in _RawOverallSummary.TEXT_FIELDS
            for issue in self._text_issues(getattr(checked, field), quotable, months)
        ]
        if nothing_found:
            issues += [f"{_SUMMARY_FIELD_NAMES[field]}: Uses suspicion wording, but nothing was found and the risk is low."
                       for field in _RawOverallSummary.TEXT_FIELDS if _SUSPICION_WORDS.search(getattr(checked, field))]
        if fit_expected and checked.occupation_fit and not checked.occupation_fit.startswith(_FIT_VERDICTS):
            issues.append(f"{_SUMMARY_FIELD_NAMES['occupation_fit']}: Does not start with a verdict "
                          "(Fits the occupation / Does not fit the occupation).")
        return self._with_issues(checked, {"overall_summary": self._log_issues(case_id, "overall_summary", issues)})

    @staticmethod
    def _without_non_sentences(raw: "_RawOverallSummary") -> "tuple[_RawOverallSummary, list[str]]":
        """Blank the lines that are not sentences, except the two the summary needs; say which, as issues."""
        blank, issues = {}, []
        for field in _RawOverallSummary.TEXT_FIELDS:
            text = getattr(raw, field)
            if not text or _is_sentence(text):
                continue
            issue = f'{_SUMMARY_FIELD_NAMES[field]}: the AI wrote "{text}" instead of a sentence'
            if field in _SUMMARY_REQUIRED_FIELDS:
                issues.append(issue + ".")
            else:
                blank[field] = ""
                issues.append(issue + ", so it is not shown.")
        return (raw.model_copy(update=blank) if blank else raw), issues

    @staticmethod
    def _check_line(check: ReviewCheck) -> str:
        months = ",".join(check.months) or "none"
        return (
            f"CHECK|{check.check}|{check.outcome}|pattern={check.pattern or 'none'}|months={months}|"
            f"insight={check.insight or 'none'}|insight_issues={' '.join(check.issues) or 'none'}|facts={check.facts}"
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
        self, request: AccountAnalysisRequest, checks: list[ReviewCheck], overall: OverallSummary | None,
        signals: RiskSignals, profile_context: CustomerProfileContext | None,
    ) -> AccountAssessment:
        # A screened check without its AI description still holds a real finding, but the case
        # must not be closed from it: staff read the figures themselves.
        all_checks_verified = all(
            check.outcome != "not_verified" and (check.check not in _SCREENED_PATTERNS or check.insight)
            for check in checks
        )
        risk_level: RiskLevel = screening_risk(signals)
        if overall is None:
            overall = OverallSummary(
                verified=False, headline=_FALLBACK_HEADLINE,
                risk_reason=f"The overall summary could not be produced; the risk is {risk_level} because "
                            f"{risk_because(signals)}.",
            )
        complete = overall.verified and all_checks_verified
        decision: Decision = (
            "no_further_review_suggested" if complete and risk_level == "low" else "further_review_suggested"
        )
        limitations = self._static_limitations(len(request.monthly_summary))
        unverified = [check.title for check in checks if check.outcome == "not_verified"]
        if unverified:
            limitations.append(AssessmentLimitation(limitation=(
                "Not verified by the LLM: " + ", ".join(unverified)
                + ". An authorised reviewer must assess these checks from the figures shown."
            )))
        if not overall.verified:
            limitations.append(AssessmentLimitation(
                limitation="The overall summary could not be produced; the risk level shown comes from the screening.",
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
        model_type: type[ModelType], max_response_tokens: int, measure: CallMeasure | None = None,
    ) -> ModelType:
        result = await self._llm_queue.submit(
            name=f"case={case_id} stage={stage}",
            operation=lambda: self._llm.complete_json(
                system_prompt=system_prompt, user_payload=prompt_input,
                response_schema=model_type.model_json_schema(), schema_name=stage.replace("-", "_"),
                max_response_tokens=max_response_tokens, measure=measure,
            ),
            measure=measure,
        )
        prepare = getattr(model_type, "prepare_payload", None)
        if prepare is not None:
            result = prepare(result)
        try:
            return model_type.model_validate(result)
        except ValidationError as exc:
            # Plain words, because this text is also the retry instruction to the model.
            raise ModelOutputError(_schema_problems(exc)) from exc

    # --- unchanged helpers (LLM retry, profile, evidence) ----------------------------

    async def _profile_context_or_none(
        self, case_id: str, profile: list[dict[str, str | None]], monthly: list[dict[str, Any]],
        measure: CallMeasure | None = None,
    ) -> "_RawProfileContext | None":
        if not profile:
            return None
        return await self._complete_with_retry(
            case_id, "profile-context", PROFILE_CONTEXT_SYSTEM_PROMPT,
            profile_context_input(profile, monthly), _RawProfileContext,
            self._settings.llm_profile_context_max_response_tokens,
            lambda value, final: self._validate_profile_context(value),
            measure,
        )

    async def _complete_with_retry(
        self, case_id: str, stage: str, system_prompt: str, prompt_input: str,
        model_type: type[ModelType], max_response_tokens: int,
        validator: Callable[[ModelType, bool], ModelType | None], measure: CallMeasure | None = None,
    ) -> ModelType:
        """Ask, validate, retry once on a format problem; ``measure`` records the call's time."""
        try:
            return await self._ask_and_validate(
                case_id, stage, system_prompt, prompt_input, model_type, max_response_tokens, validator, measure)
        except Exception:
            if measure is not None:
                measure.failed = True
            raise

    async def _ask_and_validate(
        self, case_id: str, stage: str, system_prompt: str, prompt_input: str,
        model_type: type[ModelType], max_response_tokens: int,
        validator: Callable[[ModelType, bool], ModelType | None], measure: CallMeasure | None,
    ) -> ModelType:
        """Ask, validate, retry once on a format problem.

        A validator may return a corrected copy of the result (for example after
        reconciling a field with source facts); returning None keeps the original.
        """
        try:
            result = await self._ask(case_id, stage, system_prompt, prompt_input, model_type, max_response_tokens, measure)
            corrected = validator(result, not self._settings.llm_format_retry_enabled)
            return result if corrected is None else corrected
        except LlmOutputTruncatedError as exc:
            raise ModelOutputError(f"{stage} reached its output limit; no same-budget retry was attempted.") from exc
        except (LlmOutputFormatError, ModelOutputError) as first_error:
            if not self._settings.llm_format_retry_enabled:
                raise ModelOutputError(f"{stage} output was rejected and format retry is disabled.") from first_error
            logger.warning("Context output rejected; retrying once: case=%s stage=%s error=%s", case_id, stage, first_error)
            try:
                result = await self._ask(
                    case_id, f"{stage}-format-retry", system_prompt + format_retry_suffix(first_error, list(model_type.model_fields)),
                    prompt_input, model_type, max_response_tokens, measure,
                )
                corrected = validator(result, True)
                return result if corrected is None else corrected
            except LlmOutputTruncatedError as exc:
                raise ModelOutputError(f"{stage} retry reached its output limit.") from exc
            except (LlmOutputFormatError, ModelOutputError) as second_error:
                logger.warning("Retry output rejected: case=%s stage=%s error=%s", case_id, stage, second_error)
                raise ModelOutputError(f"{stage} did not produce a verifiable result after one format retry.") from second_error

    def _inactivity_run(self, monthly: list[dict[str, Any]]) -> InactivityRun | None:
        return find_inactivity_run(monthly, min_zero_months=self._settings.dormancy_min_zero_months)

    def _validate_profile_context(self, raw: "_RawProfileContext") -> None:
        self._reject_forbidden_text(raw.profile_summary, _FORBIDDEN_PROFILE_TEXT, "profile summary")

    @staticmethod
    def _reject_forbidden_text(value: str, phrases: tuple[str, ...], label: str) -> None:
        if any(phrase in value.casefold() for phrase in phrases):
            raise ModelOutputError(f"{label} contains a prohibited missing-data phrase")

    @staticmethod
    def _split_months(value: str, label: str, min_count: int = 2, max_count: int | None = None) -> list[str]:
        """Accept source months such as ``202607,202608`` or ``M202607_M202608``, oldest first.

        This fixes punctuation and order only; it never derives a month or invents a value.
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
        return sorted(months)

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
    def _static_limitations(months: int) -> list[AssessmentLimitation]:
        return [
            AssessmentLimitation(limitation=f"Assessment uses only {months} monthly aggregates and factual linked "
                                            "customer-profile records."),
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


# How much text the code accepts from the AI before treating it as runaway output.
# Generous on purpose: the prompts ask for shorter text, and a normal answer must
# never be cut (an explanation cut mid-sentence changes what the AI said).
_INSIGHT_TEXT_LIMIT = 600
_HEADLINE_TEXT_LIMIT = 300
_SUMMARY_TEXT_LIMIT = 600


def _cap(text: str, limit: int) -> str:
    """Shorten text to ``limit`` characters at a word boundary."""
    if len(text) <= limit:
        return text
    # Leave room for "...": a cut of exactly ``limit`` plus "..." failed its own limit
    # (logged 2026-10-12: 603 characters rejected, so the whole call failed twice).
    cut = text[:limit - 3].rsplit(" ", 1)[0].rstrip(",;: ")
    return cut + "..."


class _RawAnswer(BaseModel):
    """Shared repairs for the flat JSON answer of one call. Repairs never change a decision."""

    model_config = ConfigDict(extra="forbid")
    _OUTCOME_KEYS: ClassVar[tuple[str, ...]] = ()
    _PATTERN_KEYS: ClassVar[tuple[str, ...]] = ()
    _TEXT_LIMITS: ClassVar[dict[str, int]] = {}
    # Problems the validators found, per check, for staff to see (never hidden).
    _issues: dict[str, list[str]] = PrivateAttr(default_factory=dict)
    # Checks whose part of the answer was still incomplete after the retry, with the reason.
    _unverified: dict[str, str] = PrivateAttr(default_factory=dict)

    @property
    def unverified(self) -> dict[str, str]:
        return self._unverified

    @unverified.setter
    def unverified(self, value: dict[str, str]) -> None:
        self._unverified = value

    @property
    def issues(self) -> dict[str, list[str]]:
        return self._issues

    @issues.setter
    def issues(self, value: dict[str, list[str]]) -> None:
        self._issues = value

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
                # Qwen sometimes writes its decision in the pattern field and leaves out the
                # outcome (logged 2026-10-05: "money_flow_pattern":"no_pattern_found"). The
                # decision is still Qwen's; it is only moved to the right field.
                outcome_key = key.removesuffix("_pattern") + "_outcome"
                decision = _OUTCOME_SYNONYMS.get(payload[key])
                if decision and outcome_key in cls._OUTCOME_KEYS and outcome_key not in payload:
                    logger.warning("Outcome given in %s; moved to %s.", key, outcome_key)
                    payload[outcome_key], payload[key] = decision, "none"
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
    """Call 1: two descriptions of the screening's findings, nothing to decide."""
    INSIGHT_KEYS: ClassVar[dict[ReviewCheckName, str]] = {
        "activity_and_amount_change": "activity_insight", "money_in_and_out": "money_flow_insight"}
    _TEXT_LIMITS: ClassVar[dict[str, int]] = {
        "activity_insight": _INSIGHT_TEXT_LIMIT, "money_flow_insight": _INSIGHT_TEXT_LIMIT}
    activity_insight: str = Field(default="", max_length=_INSIGHT_TEXT_LIMIT)
    money_flow_insight: str = Field(default="", max_length=_INSIGHT_TEXT_LIMIT)


class _RawInactivityBurstGaps(_RawAnswer):
    _OUTCOME_KEYS: ClassVar[tuple[str, ...]] = ("burst_gaps_outcome",)
    _PATTERN_KEYS: ClassVar[tuple[str, ...]] = ("burst_gaps_pattern",)
    _TEXT_LIMITS: ClassVar[dict[str, int]] = {
        "inactivity_insight": _INSIGHT_TEXT_LIMIT, "burst_gaps_insight": _INSIGHT_TEXT_LIMIT}
    # Fields are in the prompt's key order; the retry message lists them in this order.
    burst_gaps_insight: str = Field(default="", max_length=_INSIGHT_TEXT_LIMIT)
    burst_gaps_pattern: str = Field(default="none", max_length=40)
    burst_gaps_months: str | None = Field(default=None, max_length=200)
    burst_gaps_outcome: Outcome | None = None
    inactivity_insight: str = Field(default="", max_length=_INSIGHT_TEXT_LIMIT)


class _RawOverallSummary(_RawAnswer):
    TEXT_FIELDS: ClassVar[tuple[str, ...]] = (
        "headline", "point_1", "point_2", "point_3", "why_it_matters", "occupation_fit", "verify_1", "verify_2",
        "risk_reason",
    )
    _TEXT_LIMITS: ClassVar[dict[str, int]] = {
        field: _HEADLINE_TEXT_LIMIT if field == "headline" else _SUMMARY_TEXT_LIMIT for field in TEXT_FIELDS
    }
    headline: str = Field(default="", max_length=_HEADLINE_TEXT_LIMIT)
    point_1: str = Field(default="", max_length=_SUMMARY_TEXT_LIMIT)
    point_2: str = Field(default="", max_length=_SUMMARY_TEXT_LIMIT)
    point_3: str = Field(default="", max_length=_SUMMARY_TEXT_LIMIT)
    why_it_matters: str = Field(default="", max_length=_SUMMARY_TEXT_LIMIT)
    occupation_fit: str = Field(default="", max_length=_SUMMARY_TEXT_LIMIT)
    verify_1: str = Field(default="", max_length=_SUMMARY_TEXT_LIMIT)
    verify_2: str = Field(default="", max_length=_SUMMARY_TEXT_LIMIT)
    risk_reason: str = Field(default="", max_length=_SUMMARY_TEXT_LIMIT)


class _RawProfileContext(BaseModel):
    model_config = ConfigDict(extra="forbid")
    profile_summary: str = Field(min_length=1, max_length=650)