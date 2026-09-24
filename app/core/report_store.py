from __future__ import annotations

import html
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from app.core.models import AccountAssessment


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

    # Distinct from the app's brand red (#c8102e) so a "high risk" finding is
    # never read as merely a branded/UI element rather than a real signal.
    _SEVERITY_COLORS = {
        "low": ("#1c7a4d", "#e7f6ee"),
        "medium": ("#8a6100", "#fbf1d6"),
        "high": ("#b3261e", "#fbe9e7"),
        "critical": ("#b3261e", "#fbe9e7"),
    }
    _RISK_BORDER_COLORS = {"low": "#1c7a4d", "medium": "#8a6100", "high": "#b3261e"}

    def _render_html(self, result: AccountAssessment) -> str:
        findings = "".join(
            f"""
            <article class="finding" style="border-left-color: {self._SEVERITY_COLORS.get(finding.severity, ("#d3d6db", "#fff"))[0]}">
              <div class="finding-head">
                <h3>{html.escape(finding.category)}</h3>
                <span class="severity" style="color: {self._SEVERITY_COLORS.get(finding.severity, ("#3d4148", "#f4f3f1"))[0]}; background: {self._SEVERITY_COLORS.get(finding.severity, ("#3d4148", "#f4f3f1"))[1]}">{html.escape(finding.severity)}</span>
              </div>
              <p>{html.escape(finding.rationale)}</p>
              <ul>
                {''.join(
                    f'<li><span class="txn-ids">{html.escape(item.year_month)} &middot; '
                    f'{html.escape(item.feature)}={html.escape(item.value)}</span>'
                    f'{html.escape(item.statement)}</li>'
                    for item in finding.evidence
                )}
              </ul>
            </article>
            """
            for finding in result.findings
        ) or "<p class='muted'>No traceable material findings were returned.</p>"

        risk_border = self._RISK_BORDER_COLORS.get(result.risk_level, "#d3d6db")

        return f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Due-Diligence Assessment &middot; {html.escape(result.case_id)}</title>
<style>
:root {{ color-scheme: light; }}
body {{
  font-family: "Segoe UI", system-ui, Arial, sans-serif;
  max-width: 880px;
  margin: 48px auto;
  padding: 0 24px 64px;
  color: #3d4148;
  line-height: 1.55;
  background: #f4f3f1;
}}
h1, h2, h3 {{ color: #1c1f26; }}
header {{
  display: flex;
  align-items: baseline;
  justify-content: space-between;
  gap: 16px;
  border-bottom: 1px solid #e3e5e9;
  padding-bottom: 18px;
  margin-bottom: 24px;
}}
header h1 {{ margin: 0; font-size: 22px; }}
small {{ color: #6b7280; }}
.decision {{
  background: #fff;
  border: 1px solid #e3e5e9;
  border-left: 4px solid {risk_border};
  border-radius: 8px;
  padding: 20px 22px;
  margin-bottom: 28px;
}}
.decision h2 {{ margin-top: 0; font-size: 19px; }}
.decision p:last-child {{ margin-bottom: 0; }}
.finding {{
  background: #fff;
  border: 1px solid #e3e5e9;
  border-left: 3px solid #d3d6db;
  border-radius: 6px;
  margin: 14px 0;
  padding: 14px 16px;
}}
.finding-head {{ display: flex; justify-content: space-between; align-items: center; gap: 12px; }}
.finding h3 {{ margin: 0; font-size: 15px; }}
.finding .confidence {{ color: #6b7280; font-size: 12.5px; margin: 6px 0 0; }}
.finding ul {{ margin: 8px 0 0; padding: 0; list-style: none; font-size: 13.5px; }}
.finding li {{ border-top: 1px solid #e3e5e9; padding: 6px 0; }}
.finding li:first-child {{ border-top: 0; }}
.txn-ids {{ font-family: Consolas, monospace; font-weight: 700; color: #1c1f26; margin-right: 8px; }}
.severity {{
  padding: 3px 9px;
  border-radius: 3px;
  font-size: 10.5px;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: .3px;
  white-space: nowrap;
}}
.muted {{ color: #6b7280; }}
ul.plain {{ padding-left: 20px; }}
</style>
</head>
<body>
<header>
  <h1>Transaction Due-Diligence Assessment</h1>
  <small>Case: {html.escape(result.case_id)} &middot; Account: {html.escape(result.acct_num)}<br>Generated: {html.escape(result.generated_at.isoformat())}</small>
</header>
<section class="decision">
  <h2>Recommendation: {html.escape(result.decision.replace("_", " ").title())} ({html.escape(result.risk_level.title())} risk)</h2>
</section>
<h2>Executive summary</h2>
<p>{html.escape(result.executive_summary)}</p>
<p><small>{result.months_reviewed} month(s) reviewed in this account's summary.</small></p>
<h2>Material findings</h2>
{findings}
<h2>Limitations</h2>
<ul class="plain">{''.join(f'<li>{html.escape(item)}</li>' for item in result.limitations) or '<li class="muted">None recorded.</li>'}</ul>
</body>
</html>"""