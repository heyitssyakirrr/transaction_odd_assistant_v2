from __future__ import annotations

import html
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from app.core.models import AccountAssessment, EvidenceTable, ReviewCheck


@dataclass(frozen=True)
class SavedReport:
    html_name: str
    json_name: str


class ReportStore:
    """Stores local, auditable analysis reports."""

    def __init__(self, directory: Path) -> None:
        self._directory = directory
        self._directory.mkdir(parents=True, exist_ok=True)

    def save(self, result: AccountAssessment) -> SavedReport:
        # A timestamp alone collides when the same customer CSV is processed
        # twice in one second. The random suffix makes report names safe for
        # concurrent requests without exposing source data in the filename.
        report_id = f"{result.generated_at.strftime('%Y%m%d-%H%M%S')}-{uuid4().hex[:12]}"
        safe_case_id = "".join(
            char if char.isalnum() or char in "-_" else "-"
            for char in result.case_id
        )

        basename = f"{safe_case_id}-{report_id}"
        json_name = f"{basename}.json"
        html_name = f"{basename}.html"

        self._atomic_write(json_name, result.model_dump_json(indent=2))
        self._atomic_write(html_name, self._render_html(result))

        return SavedReport(html_name=html_name, json_name=json_name)

    def resolve(self, report_name: str) -> Path | None:
        candidate = (self._directory / Path(report_name).name).resolve()

        if candidate.parent != self._directory.resolve():
            return None

        return candidate if candidate.exists() else None

    def _atomic_write(self, filename: str, content: str) -> None:
        """Publish a report only after its complete content is on disk."""
        temp_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=self._directory,
                prefix=f".{filename}.",
                suffix=".tmp",
                delete=False,
            ) as stream:
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
                temp_path = Path(stream.name)
            os.replace(temp_path, self._directory / filename)
        finally:
            if temp_path is not None and temp_path.exists():
                temp_path.unlink(missing_ok=True)

    # Risk colours are distinct from the app's brand red (#c8102e), so "high risk" is
    # never read as page branding.
    _RISK_COLORS = {"low": "#1c7a4d", "medium": "#8a6100", "high": "#b3261e"}
    _OUTCOME_LABELS = {
        "pattern_found": "Pattern found", "no_pattern_found": "No pattern found", "not_verified": "Not verified",
    }
    _OUTCOME_COLORS = {
        "pattern_found": ("#8a6100", "#fbf1d6"), "no_pattern_found": ("#1c7a4d", "#e7f6ee"),
        "not_verified": ("#b3261e", "#fbe9e7"),
    }
    _DECISION_LABELS = {
        "further_review_suggested": "Further review suggested",
        "no_further_review_suggested": "No further review suggested",
    }

    @staticmethod
    def _table(table: EvidenceTable | None, css_class: str = "") -> str:
        if table is None or not table.rows:
            return ""
        head = "".join(f"<th>{html.escape(column)}</th>" for column in table.columns)
        body = "".join(
            f'<tr class="{"sel" if index in table.highlighted_rows else ""}">'
            + "".join(f"<td>{html.escape(cell)}</td>" for cell in row) + "</tr>"
            for index, row in enumerate(table.rows)
        )
        return f'<table class="{css_class}"><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>'

    @staticmethod
    def _insight_note(check: ReviewCheck) -> str:
        """Why a check has no AI explanation; the same wording is used in app/static/results.js."""
        if check.insight_status == "hidden":
            return "The AI's explanation was hidden because it did not fit the figures. Use the table below."
        if check.outcome == "pattern_found":
            return "The AI did not provide an explanation for this check. Use the table below."
        return ""

    def _check(self, check: ReviewCheck) -> str:
        color, tint = self._OUTCOME_COLORS[check.outcome]
        pattern = f'<span class="tag">{html.escape(check.pattern)}</span>' if check.pattern else ""
        months = f'<p class="months">Months: <strong>{" &rarr; ".join(map(html.escape, check.months))}</strong></p>' if check.months else ""
        if check.insight:
            insight = f'<p class="insight">{html.escape(check.insight)}</p>'
        elif note := self._insight_note(check):
            insight = f"<p class='muted'><em>{note}</em></p>"
        else:
            insight = ""
        return f"""
<article class="card" style="border-left-color: {color}">
  <div class="card-head"><h3>{html.escape(check.title)}</h3>
    <div>{pattern}<span class="chip" style="color:{color};background:{tint}">{self._OUTCOME_LABELS[check.outcome]}</span></div></div>
  {months}{insight}
  <p class="facts"><b>KEY FIGURES</b> {html.escape(check.facts)}</p>
  {self._table(check.table)}
</article>"""

    def _render_html(self, result: AccountAssessment) -> str:
        overall = result.overall
        risk_color = self._RISK_COLORS.get(result.risk_level, "#d3d6db")
        points = "".join(f"<li>{html.escape(point)}</li>" for point in overall.points) or \
            "<li class='muted'>No summary points are available. Review each check below.</li>"
        why = f"<div class='block'><h4>Why it matters</h4><p>{html.escape(overall.why_it_matters)}</p></div>" if overall.why_it_matters else ""
        verify = (
            "<div class='block'><h4>What to verify first</h4><ol>"
            + "".join(f"<li>{html.escape(item)}</li>" for item in overall.verify_first) + "</ol></div>"
        ) if overall.verify_first else ""
        reason = (
            f"<p class='reason'><b>Why {html.escape(result.risk_level)} risk:</b> {html.escape(overall.risk_reason)}</p>"
            if overall.risk_reason else ""
        )
        note = (
            "Part of this result could not be verified. Review the checks below before deciding; do not close the case from this result alone."
            if result.status == "needs_review"
            else "AI decision support only. An authorised reviewer remains responsible for the case decision."
        )
        profile = ""
        if result.customer_profile_context is not None:
            context = result.customer_profile_context
            profile = f"""
<h2>{html.escape(context.title)}</h2>
<article class="card"><p class="insight">{html.escape(context.summary)}</p>{self._table(context.table, "kv")}</article>"""
        checks = "".join(self._check(check) for check in result.review_checks)
        limitations = "".join(f"<li>{html.escape(item.limitation)}</li>" for item in result.limitations) or "<li class='muted'>None recorded.</li>"

        return f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Due-Diligence Assessment &middot; {html.escape(result.case_id)}</title>
<style>
:root {{ color-scheme: light; }}
body {{ font-family: "Segoe UI", system-ui, Arial, sans-serif; max-width: 960px; margin: 40px auto; padding: 0 24px 64px;
  color: #3d4148; line-height: 1.55; background: #f4f3f1; }}
h1, h2, h3, h4 {{ color: #1c1f26; }}
header {{ display: flex; align-items: baseline; justify-content: space-between; gap: 16px; border-bottom: 1px solid #e3e5e9;
  padding-bottom: 16px; margin-bottom: 22px; }}
header h1 {{ margin: 0; font-size: 22px; }}
small, .muted {{ color: #6b7280; }}
.decision {{ background: #fff; border: 1px solid #e3e5e9; border-left: 5px solid {risk_color}; border-radius: 8px; padding: 20px 22px; }}
.decision .risk {{ display: inline-block; padding: 4px 10px; border-radius: 4px; font-size: 12px; font-weight: 700;
  color: {risk_color}; background: #fff; border: 1px solid {risk_color}; }}
.decision h2 {{ margin: 10px 0 6px; font-size: 21px; }}
.headline {{ margin: 0 0 8px; font-size: 16px; font-weight: 600; color: #1c1f26; }}
ul.points {{ padding-left: 20px; font-size: 15px; color: #1c1f26; }}
ul.points li {{ margin: 6px 0; }}
.blocks {{ display: grid; grid-template-columns: 1fr 1fr; gap: 14px; }}
.block {{ background: #fff; border: 1px solid #e3e5e9; border-radius: 6px; padding: 12px 16px; }}
.block h4 {{ margin: 0 0 6px; font-size: 12px; text-transform: uppercase; letter-spacing: .4px; color: #6b7280; }}
.block p, .block ol {{ margin: 0; }}
.reason {{ background: #fff; border-radius: 6px; padding: 10px 14px; border: 1px solid #e3e5e9; }}
.card {{ background: #fff; border: 1px solid #e3e5e9; border-left: 4px solid #d3d6db; border-radius: 6px; margin: 14px 0; padding: 16px 18px; }}
.card-head {{ display: flex; justify-content: space-between; align-items: center; gap: 12px; flex-wrap: wrap; }}
.card h3 {{ margin: 0; font-size: 15.5px; }}
.chip, .tag {{ padding: 3px 9px; border-radius: 999px; font-size: 11.5px; font-weight: 700; margin-left: 6px; }}
.tag {{ background: #f4f3f1; border: 1px solid #e3e5e9; }}
.insight {{ font-size: 14.5px; color: #1c1f26; }}
.facts, .months {{ font-size: 13px; }}
.months {{ color: #6b7280; }}
.facts b {{ font-size: 11px; color: #6b7280; letter-spacing: .4px; margin-right: 6px; }}
table {{ width: 100%; border-collapse: collapse; font-size: 13.5px; margin-top: 10px; font-variant-numeric: tabular-nums; }}
th {{ text-align: right; padding: 7px 10px; background: #f4f3f1; border-bottom: 2px solid #d3d6db; color: #1c1f26; vertical-align: bottom; }}
td {{ text-align: right; padding: 6px 10px; border-bottom: 1px solid #e3e5e9; white-space: nowrap; }}
th:first-child, td:first-child {{ text-align: left; }}
tr.sel td {{ font-weight: 700; color: #1c1f26; background: #fbf1d6; }}
table.kv td, table.kv th {{ text-align: left; white-space: normal; }}
table.kv td:first-child {{ width: 38%; color: #6b7280; }}
</style>
</head>
<body>
<header>
  <h1>Transaction Due-Diligence Assessment</h1>
  <small>Case: {html.escape(result.case_id)} &middot; Account: {html.escape(result.acct_num)}<br>Generated: {result.generated_at.strftime('%Y-%m-%d %H:%M UTC')}</small>
</header>
<section class="decision">
  <span class="risk">{html.escape(result.risk_level.upper())} RISK</span>
  <h2>{self._DECISION_LABELS[result.decision]}</h2>
  <p class="headline">{html.escape(overall.headline)}</p>
  <p class="muted">{note}</p>
</section>
<h2>Summary</h2>
<ul class="points">{points}</ul>
<div class="blocks">{why}{verify}</div>
{reason}
<h2>Review checks</h2>
<p class="muted">Explanations are written by the AI; every figure in the tables comes from the uploaded CSV. Months the AI selected are in bold.
{result.months_reviewed} month(s) reviewed; {result.profile_records_matched} linked profile record(s) available.</p>
{checks}
{profile}
<h2>Limitations</h2>
<ul>{limitations}</ul>
</body>
</html>"""