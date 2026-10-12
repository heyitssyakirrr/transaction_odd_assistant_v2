const RESULT_STORAGE_KEY = "dd:last-result";
const BATCH_POLL_MS = 5000;
const FILE_STATUS_LABELS = {
  queued: "Waiting",
  analysing: "Analysing",
  done: "Ready",
  failed: "Failed",
};

const OUTCOME_LABELS = {
  pattern_found: "Pattern found",
  no_pattern_found: "No pattern found",
  not_verified: "Not verified",
};
// "not_assessed": the AI could not decide the risk because a check it depends on was not verified.
const RISK_LABELS = { low: "LOW RISK", medium: "MEDIUM RISK", high: "HIGH RISK", not_assessed: "RISK NOT ASSESSED" };
const RISK_REASON_LABELS = { not_assessed: "Why the risk was not assessed:" };

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
  occupationBlock: document.querySelector("#occupation-block"),
  occupationFit: document.querySelector("#occupation-fit"),
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
  emptyStateText: document.querySelector("#empty-state-text"),
  batchBar: document.querySelector("#batch-bar"),
  batchPosition: document.querySelector("#batch-position"),
  batchFile: document.querySelector("#batch-file"),
  batchFileStatus: document.querySelector("#batch-file-status"),
  batchProgress: document.querySelector("#batch-progress"),
  batchPlaceholder: document.querySelector("#batch-placeholder"),
  batchPlaceholderText: document.querySelector("#batch-placeholder-text"),
  prevReport: document.querySelector("#prev-report"),
  nextReport: document.querySelector("#next-report"),
};

// Batch mode: /results?batch=<id>&page=<n>. One report per page; the batch is polled
// until every file has finished, and a page is drawn once its report is ready.
const batchState = { id: null, data: null, page: 1, shown: null };

init();

function init() {
  const params = new URLSearchParams(window.location.search);
  const batchId = params.get("batch");
  if (batchId) {
    initBatch(batchId, Number.parseInt(params.get("page"), 10) || 1);
    return;
  }
  initSingleResult();
}

function initBatch(batchId, page) {
  batchState.id = batchId;
  batchState.page = page;
  els.emptyState.classList.add("hidden");
  els.prevReport.addEventListener("click", () => goToPage(batchState.page - 1));
  els.nextReport.addEventListener("click", () => goToPage(batchState.page + 1));
  pollBatch();
}

async function pollBatch() {
  try {
    const response = await fetch(`/v1/accounts/batches/${encodeURIComponent(batchState.id)}`);
    const body = await response.json().catch(() => null);
    if (!response.ok) {
      showBatchError(body?.detail || `Could not load this upload (HTTP ${response.status}).`);
      return;
    }
    batchState.data = body;
    batchState.page = Math.min(Math.max(batchState.page, 1), body.total);
    showBatchPage();
    if (!body.finished) setTimeout(pollBatch, BATCH_POLL_MS);
  } catch {
    // A dropped connection should not stop the page; try again shortly.
    setTimeout(pollBatch, BATCH_POLL_MS);
  }
}

function goToPage(page) {
  if (!batchState.data || page < 1 || page > batchState.data.total) return;
  batchState.page = page;
  const url = new URL(window.location.href);
  url.searchParams.set("page", String(page));
  window.history.replaceState(null, "", url);
  showBatchPage();
  window.scrollTo({ top: 0 });
}

function showBatchPage() {
  const { data, page } = batchState;
  const file = data.files[page - 1];

  els.batchBar.classList.remove("hidden");
  els.batchPosition.textContent = `Report ${page} of ${data.total}`;
  els.batchFile.textContent = file.filename;
  els.batchFileStatus.textContent = FILE_STATUS_LABELS[file.status] || file.status;
  els.batchFileStatus.className = `file-status status-${file.status}`;
  els.prevReport.disabled = page <= 1;
  els.nextReport.disabled = page >= data.total;

  const counts = data.counts;
  els.batchProgress.textContent = data.finished
    ? `All ${data.total} files finished: ${counts.done} ready, ${counts.failed} failed.`
    : `${counts.done} ready \u00b7 ${counts.analysing} analysing \u00b7 ${counts.queued} waiting \u00b7 ${counts.failed} failed. `
      + "Files are analysed one at a time.";
  els.batchProgress.classList.remove("hidden");

  if (file.status === "done" && file.result) {
    const key = `${page}:done`;
    if (batchState.shown !== key) {  // draw once, so polling does not reset the page
      renderResult(file.result);
      batchState.shown = key;
    }
    els.batchPlaceholder.classList.add("hidden");
    els.result.classList.remove("hidden");
    return;
  }

  batchState.shown = null;
  els.result.classList.add("hidden");
  els.batchPlaceholderText.textContent = placeholderText(file);
  els.batchPlaceholder.classList.remove("hidden");
}

function placeholderText(file) {
  if (file.status === "failed") return `This file could not be analysed: ${file.error}`;
  if (file.status === "analysing") return "Analysing this account now. This usually takes a few minutes; the report appears here when it is ready.";
  return "Waiting in the queue. Files are analysed one at a time, so this one starts after the files before it.";
}

function showBatchError(message) {
  els.batchBar.classList.add("hidden");
  els.batchProgress.classList.add("hidden");
  els.batchPlaceholder.classList.add("hidden");
  els.result.classList.add("hidden");
  els.emptyStateText.textContent = message;
  els.emptyState.classList.remove("hidden");
}

function initSingleResult() {
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
  els.riskBadge.textContent = RISK_LABELS[data.risk_level] || `${data.risk_level.toUpperCase()} RISK`;
  els.riskBadge.className = `risk-badge risk-${data.risk_level}`;
  els.decision.textContent = DECISION_LABELS[data.decision] || data.decision;
  els.headline.textContent = data.overall.headline;
  els.decisionRationale.textContent = data.status === "needs_review"
    ? "Part of this result could not be verified. Review the checks below before deciding; do not close the case from this result alone."
    : "AI decision support only. An authorised reviewer remains responsible for the case decision.";

  els.acctNum.textContent = data.acct_num;
  els.monthsReviewed.textContent = data.months_reviewed;
  els.profileRecordsMatched.textContent = data.profile_records_matched ?? 0;
  els.riskLevel.textContent = data.risk_level === "not_assessed" ? "NOT ASSESSED" : data.risk_level.toUpperCase();

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

  toggle(els.occupationBlock, Boolean(overall.occupation_fit));
  els.occupationFit.textContent = overall.occupation_fit || "";

  const verify = overall.verify_first || [];
  toggle(els.verifyBlock, verify.length > 0);
  els.verifyFirst.innerHTML = verify.map((item) => `<li>${escapeHtml(item)}</li>`).join("");

  els.riskReason.innerHTML = (overall.risk_reason
    ? `<strong>${escapeHtml(RISK_REASON_LABELS[riskLevel] || `Why ${riskLevel} risk:`)}</strong> ${escapeHtml(overall.risk_reason)}`
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