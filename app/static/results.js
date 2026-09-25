const RESULT_STORAGE_KEY = "dd:last-result";
const SEVERITY_ORDER = { critical: 0, high: 1, medium: 2, low: 3 };

const els = {
  emptyState: document.querySelector("#empty-state"),
  result: document.querySelector("#result"),
  caseId: document.querySelector("#case-id"),
  generatedAt: document.querySelector("#generated-at"),
  decisionCard: document.querySelector("#decision-card"),
  decision: document.querySelector("#decision"),
  riskBadge: document.querySelector("#risk-badge"),
  acctNum: document.querySelector("#acct-num"),
  monthsReviewed: document.querySelector("#months-reviewed"),
  riskLevel: document.querySelector("#risk-level"),
  summary: document.querySelector("#summary"),
  monthlyComparison: document.querySelector("#monthly-comparison"),
  profileNotes: document.querySelector("#profile-notes"),
  findings: document.querySelector("#findings"),
  findingsCount: document.querySelector("#findings-count"),
  limitations: document.querySelector("#limitations"),
  reportHtmlLink: document.querySelector("#report-html-link"),
  reportJsonLink: document.querySelector("#report-json-link"),
};

init();

function init() {
  const raw = sessionStorage.getItem(RESULT_STORAGE_KEY);
  const data = raw ? safeParse(raw) : null;

  if (!data) {
    els.emptyState.classList.remove("hidden");
    els.result.classList.add("hidden");
    return;
  }

  renderResult(data);
  els.emptyState.classList.add("hidden");
  els.result.classList.remove("hidden");
}

function safeParse(raw) {
  try {
    return JSON.parse(raw);
  } catch {
    return null;
  }
}

function renderResult(data) {
  els.caseId.textContent = data.case_id;
  els.generatedAt.textContent = data.generated_at
    ? `Generated ${new Date(data.generated_at).toLocaleString()}`
    : "";

  els.decisionCard.className = `decision-card risk-${data.risk_level}`;
  els.decision.textContent = formatEnum(data.decision);

  els.riskBadge.textContent = `${data.risk_level.toUpperCase()} RISK`;
  els.riskBadge.className = `risk-badge risk-${data.risk_level}`;

  els.acctNum.textContent = data.acct_num;
  els.monthsReviewed.textContent = data.months_reviewed;
  els.riskLevel.textContent = data.risk_level.toUpperCase();

  els.summary.textContent = data.executive_summary;

  renderMonthlyComparison(data.monthly_comparison || []);
  renderProfileNotes(data.profile_notes || []);
  renderFindings(data.findings || []);

  els.limitations.innerHTML = (data.limitations || [])
    .map((item) => `<li>${escapeHtml(item)}</li>`)
    .join("") || "<li class='muted'>None recorded.</li>";

  els.reportHtmlLink.href = data.report_html;
  els.reportJsonLink.href = data.report_json;
}

function renderMonthlyComparison(notes) {
  if (!notes.length) {
    els.monthlyComparison.innerHTML = "<li class='muted'>No feature stood out from this account's own 6-month pattern.</li>";
    return;
  }
  els.monthlyComparison.innerHTML = notes
    .map((note) => {
      const months = (note.notable_months || []).map(escapeHtml).join(", ");
      return `
        <li>
          <span class="plain-list-label">${escapeHtml(formatEnum(note.feature))}</span>
          <span>${escapeHtml(note.pattern_summary)}</span>
          ${months ? `<span class="transaction-ids">${months}</span>` : ""}
        </li>`;
    })
    .join("");
}

function renderProfileNotes(notes) {
  if (!notes.length) {
    els.profileNotes.innerHTML = "<li class='muted'>No profile change recorded in the supplied history.</li>";
    return;
  }
  els.profileNotes.innerHTML = notes
    .map((note) => {
      const when = note.change_dttm ? new Date(note.change_dttm).toLocaleDateString() : "date not recorded";
      const flag = note.coincides_with_txn_pattern
        ? "<span class='severity medium'>Coincides with comparison above</span>"
        : "";
      return `
        <li>
          <span class="plain-list-label">${escapeHtml(formatEnum(note.field))}</span>
          <span>${escapeHtml(note.change_summary)} (${escapeHtml(when)})</span>
          ${flag}
        </li>`;
    })
    .join("");
}

function renderFindings(findings) {
  els.findingsCount.textContent = findings.length
    ? `${findings.length} finding${findings.length === 1 ? "" : "s"}`
    : "";

  if (!findings.length) {
    els.findings.innerHTML =
      "<p class='muted'>No material finding with a verified year_month citation was returned.</p>";
    return;
  }

  const sorted = [...findings].sort(
    (a, b) => (SEVERITY_ORDER[a.severity] ?? 9) - (SEVERITY_ORDER[b.severity] ?? 9)
  );

  els.findings.innerHTML = sorted
    .map((finding) => {
      const evidenceItems = finding.evidence
        .map(
          (item) => `
            <li>
              <span class="transaction-ids">${escapeHtml(item.year_month)} &middot; ${escapeHtml(item.feature)}=${escapeHtml(item.value)}</span>
              <span>${escapeHtml(item.statement)}</span>
            </li>`
        )
        .join("");

      return `
        <article class="finding severity-${escapeHtml(finding.severity)}">
          <div class="finding-header">
            <h4>${escapeHtml(formatEnum(finding.category))}</h4>
            <span class="severity ${escapeHtml(finding.severity)}">${escapeHtml(finding.severity)}</span>
          </div>
          <p>${escapeHtml(finding.rationale)}</p>
          <h5>Evidence</h5>
          <ul>${evidenceItems}</ul>
        </article>`;
    })
    .join("");
}

function formatEnum(value) {
  return value.replaceAll("_", " ").replace(/\b\w/g, (char) => char.toUpperCase());
}

function escapeHtml(value) {
  const element = document.createElement("div");
  element.textContent = value ?? "";
  return element.innerHTML;
}