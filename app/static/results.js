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
  profileRecordsMatched: document.querySelector("#profile-records-matched"),
  riskLevel: document.querySelector("#risk-level"),
  summary: document.querySelector("#summary"),
  reviewChecks: document.querySelector("#review-checks"),
  findings: document.querySelector("#findings"),
  findingsCount: document.querySelector("#findings-count"),
  reviewerQuestions: document.querySelector("#reviewer-questions"),
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
  els.profileRecordsMatched.textContent = data.profile_records_matched ?? 0;
  els.riskLevel.textContent = data.risk_level.toUpperCase();

  els.summary.textContent = data.executive_summary;

  renderReviewChecks(data.review_checks || []);
  renderFindings(data.findings || []);

  els.reviewerQuestions.innerHTML = (data.reviewer_questions || [])
    .map((item) => `<li>${escapeHtml(item.question)}${renderInlineEvidence(item.evidence || [])}</li>`)
    .join("") || "<li class='muted'>No additional reviewer question was returned.</li>";

  els.limitations.innerHTML = (data.limitations || [])
    .map((item) => `<li>${escapeHtml(item.limitation)}${renderInlineEvidence(item.evidence || [])}</li>`)
    .join("") || "<li class='muted'>None recorded.</li>";

  els.reportHtmlLink.href = data.report_html;
  els.reportJsonLink.href = data.report_json;
}

function renderReviewChecks(checks) {
  if (!checks.length) {
    els.reviewChecks.innerHTML = "<p class='muted'>Review coverage was not returned.</p>";
    return;
  }

  els.reviewChecks.innerHTML = checks.map((check) => {
    const outcomeClass = check.outcome === "observed" ? "medium" : "low";
    const evidenceItems = renderEvidenceItems(check.evidence || []);
    return `
      <article class="finding severity-${outcomeClass}">
        <div class="finding-header">
          <h4>${escapeHtml(formatEnum(check.check))}</h4>
          <span class="severity ${outcomeClass}">${escapeHtml(formatEnum(check.outcome))}</span>
        </div>
        <p>${escapeHtml(check.rationale)}</p>
        ${evidenceItems ? `<h5>Evidence reviewed</h5><ul>${evidenceItems}</ul>` : ""}
      </article>`;
  }).join("");
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
      const evidenceItems = renderEvidenceItems(finding.evidence || []);

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

function renderEvidenceItems(evidence) {
  return evidence.map((item) =>
    `<li><span class="transaction-ids">${escapeHtml(item.label)}=${escapeHtml(item.value)}</span></li>`
  ).join("");
}

function renderInlineEvidence(evidence) {
  if (!evidence.length) return "";
  return `<br><span class="transaction-ids">${evidence.map((item) =>
    `${escapeHtml(item.label)}=${escapeHtml(item.value)}`
  ).join("; ")}</span>`;
}

function formatEnum(value) {
  return value.replaceAll("_", " ").replace(/\b\w/g, (char) => char.toUpperCase());
}

function escapeHtml(value) {
  const element = document.createElement("div");
  element.textContent = value ?? "";
  return element.innerHTML;
}
