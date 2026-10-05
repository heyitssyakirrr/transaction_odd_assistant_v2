const RESULT_STORAGE_KEY = "dd:last-result";

const OUTCOME_LABELS = {
  pattern_found: "Pattern found",
  no_pattern_found: "No pattern found",
  not_verified: "Not verified",
};
const DECISION_LABELS = {
  further_review_suggested: "Further review suggested",
  no_further_review_suggested: "No further review suggested",
};

const els = {
  emptyState: document.querySelector("#empty-state"),
  result: document.querySelector("#result"),
  caseId: document.querySelector("#case-id"),
  generatedAt: document.querySelector("#generated-at"),
  decisionCard: document.querySelector("#decision-card"),
  decision: document.querySelector("#decision"),
  headline: document.querySelector("#headline"),
  decisionRationale: document.querySelector("#decision-rationale"),
  riskBadge: document.querySelector("#risk-badge"),
  acctNum: document.querySelector("#acct-num"),
  monthsReviewed: document.querySelector("#months-reviewed"),
  profileRecordsMatched: document.querySelector("#profile-records-matched"),
  riskLevel: document.querySelector("#risk-level"),
  summaryPoints: document.querySelector("#summary-points"),
  whyBlock: document.querySelector("#why-block"),
  whyItMatters: document.querySelector("#why-it-matters"),
  verifyBlock: document.querySelector("#verify-block"),
  verifyFirst: document.querySelector("#verify-first"),
  riskReason: document.querySelector("#risk-reason"),
  checksCount: document.querySelector("#checks-count"),
  reviewChecks: document.querySelector("#review-checks"),
  customerProfileSection: document.querySelector("#customer-profile-section"),
  customerProfileTitle: document.querySelector("#customer-profile-title"),
  customerProfileContext: document.querySelector("#customer-profile-context"),
  limitationsTitle: document.querySelector("#limitations-title"),
  limitations: document.querySelector("#limitations"),
  reportHtmlLink: document.querySelector("#report-html-link"),
  reportJsonLink: document.querySelector("#report-json-link"),
};

init();

function init() {
  const raw = sessionStorage.getItem(RESULT_STORAGE_KEY);
  const data = raw ? safeParse(raw) : null;

  if (!data || !data.overall) {
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
  els.riskBadge.textContent = `${data.risk_level.toUpperCase()} RISK`;
  els.riskBadge.className = `risk-badge risk-${data.risk_level}`;
  els.decision.textContent = DECISION_LABELS[data.decision] || data.decision;
  els.headline.textContent = data.overall.headline;
  els.decisionRationale.textContent = data.status === "needs_review"
    ? "Part of this result could not be verified. Review the checks below before deciding; do not close the case from this result alone."
    : "AI decision support only. An authorised reviewer remains responsible for the case decision.";

  els.acctNum.textContent = data.acct_num;
  els.monthsReviewed.textContent = data.months_reviewed;
  els.profileRecordsMatched.textContent = data.profile_records_matched ?? 0;
  els.riskLevel.textContent = data.risk_level.toUpperCase();

  renderSummary(data.overall, data.risk_level);
  renderReviewChecks(data.review_checks || []);
  renderCustomerProfile(data.customer_profile_context);
  renderLimitations(data.limitations || []);

  els.reportHtmlLink.href = data.report_html;
  els.reportJsonLink.href = data.report_json;
}

function renderSummary(overall, riskLevel) {
  const points = overall.points || [];
  els.summaryPoints.innerHTML = points.length
    ? points.map((point) => `<li>${escapeHtml(point)}</li>`).join("")
    : "<li class='muted'>No summary points are available. Review each check below.</li>";

  toggle(els.whyBlock, Boolean(overall.why_it_matters));
  els.whyItMatters.textContent = overall.why_it_matters || "";

  const verify = overall.verify_first || [];
  toggle(els.verifyBlock, verify.length > 0);
  els.verifyFirst.innerHTML = verify.map((item) => `<li>${escapeHtml(item)}</li>`).join("");

  els.riskReason.innerHTML = (overall.risk_reason
    ? `<strong>Why ${escapeHtml(riskLevel)} risk:</strong> ${escapeHtml(overall.risk_reason)}`
    : "") + renderIssues(overall.issues);
}

function renderReviewChecks(checks) {
  const found = checks.filter((check) => check.outcome === "pattern_found").length;
  els.checksCount.textContent = `${found} of ${checks.length} with a pattern found`;
  els.reviewChecks.innerHTML = checks.map(renderCheck).join("")
    || "<p class='muted'>Review checks were not returned.</p>";
}

function renderCheck(check) {
  const months = check.months && check.months.length
    ? `<p class="check-months">Months: <strong>${check.months.map(escapeHtml).join(" &rarr; ")}</strong></p>`
    : "";
  const pattern = check.pattern ? `<span class="pattern-tag">${escapeHtml(check.pattern)}</span>` : "";
  const insight = (check.insight
    ? `<p class="check-insight">${escapeHtml(check.insight)}</p>`
    : check.outcome === "pattern_found"
      ? "<p class='check-note'>The AI did not provide an explanation for this check. Use the table below.</p>"
      : "") + renderIssues(check.issues);
  return `
    <article class="check-card outcome-${escapeHtml(check.outcome)}">
      <div class="check-head">
        <h4>${escapeHtml(check.title)}</h4>
        <div class="check-tags">
          ${pattern}
          <span class="outcome-chip outcome-${escapeHtml(check.outcome)}">${escapeHtml(OUTCOME_LABELS[check.outcome] || check.outcome)}</span>
        </div>
      </div>
      ${months}
      ${insight}
      <p class="check-facts"><span>Key figures</span> ${escapeHtml(check.facts)}</p>
      ${renderTable(check.table)}
    </article>`;
}

// Where the AI's answer does not match the CSV; the same wording is used in app/core/report_store.py.
function renderIssues(issues) {
  if (!issues || !issues.length) return "";
  return `<div class="check-issues"><strong>Check this AI answer against the table:</strong>
    <ul>${issues.map((issue) => `<li>${escapeHtml(issue)}</li>`).join("")}</ul></div>`;
}

function renderCustomerProfile(context) {
  if (!context) {
    els.customerProfileSection.classList.add("hidden");
    return;
  }
  els.customerProfileSection.classList.remove("hidden");
  els.customerProfileTitle.textContent = context.title || "Customer profile vs activity";
  els.customerProfileContext.innerHTML = `
    <article class="check-card">
      <p class="check-insight">${escapeHtml(context.summary)}</p>
      ${renderTable(context.table, "profile-table")}
    </article>`;
}

function renderLimitations(limitations) {
  els.limitationsTitle.textContent = `Limitations (${limitations.length})`;
  els.limitations.innerHTML = limitations
    .map((item) => `<li>${escapeHtml(item.limitation)}</li>`)
    .join("") || "<li class='muted'>None recorded.</li>";
}

function renderTable(table, extraClass = "") {
  if (!table || !table.columns || !table.rows || !table.rows.length) return "";
  const highlighted = new Set(table.highlighted_rows || []);
  const head = table.columns.map((column) => `<th>${escapeHtml(column)}</th>`).join("");
  const body = table.rows.map((row, index) => `
    <tr class="${highlighted.has(index) ? "is-selected" : ""}">
      ${row.map((cell) => `<td>${escapeHtml(cell)}</td>`).join("")}
    </tr>`).join("");
  return `
    <div class="table-wrap">
      <table class="evidence-table ${extraClass}">
        <thead><tr>${head}</tr></thead>
        <tbody>${body}</tbody>
      </table>
    </div>`;
}

function toggle(element, visible) {
  element.classList.toggle("hidden", !visible);
}

function escapeHtml(value) {
  const element = document.createElement("div");
  element.textContent = value ?? "";
  return element.innerHTML;
}