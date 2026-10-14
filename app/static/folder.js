// Upload page — "Read from folder": analyse accounts straight from the server folder.
//
// The folder itself is fixed on the server (ACCOUNT_FOLDER_PATH); staff choose the
// citizenship and a range of the matching files, by file name: from 1 to 5 is the first
// five, from 6 to 10 the next five. They are queued as one batch and followed on the
// same results page as an upload.

const SECONDS_PER_ACCOUNT = 80; // rough guide from UAT runs, for the time estimate only
const DEFAULT_RANGE_SIZE = 5;   // a small first run; change "To account" for more

const folderEls = {
  switchButtons: document.querySelectorAll(".source-option"),
  uploadForm: document.querySelector("#analysis-form"),
  folderForm: document.querySelector("#folder-form"),
  loading: document.querySelector("#folder-loading"),
  panel: document.querySelector("#folder-panel"),
  path: document.querySelector("#folder-path"),
  total: document.querySelector("#folder-total"),
  start: document.querySelector("#range-start"),
  end: document.querySelector("#range-end"),
  countHint: document.querySelector("#count-hint"),
  citizenCodes: document.querySelector("#citizen-codes"),
  unreadable: document.querySelector("#folder-unreadable"),
  error: document.querySelector("#folder-error"),
  submit: document.querySelector("#folder-submit"),
  submitLabel: document.querySelector("#folder-submit-label"),
};

const folderState = { loaded: false, summary: null };

folderEls.switchButtons.forEach((button) => {
  button.addEventListener("click", () => showSource(button.dataset.source));
});
folderEls.folderForm.addEventListener("change", (event) => {
  if (event.target.name === "citizenship") updateRange(true);
});
folderEls.start.addEventListener("input", () => updateRange(false));
folderEls.end.addEventListener("input", () => updateRange(false));
folderEls.folderForm.addEventListener("submit", startFolderBatch);

function showSource(source) {
  folderEls.switchButtons.forEach((button) => {
    const active = button.dataset.source === source;
    button.classList.toggle("is-active", active);
    button.setAttribute("aria-selected", String(active));
  });
  folderEls.uploadForm.classList.toggle("hidden", source !== "upload");
  folderEls.folderForm.classList.toggle("hidden", source !== "folder");
  if (source === "folder" && !folderState.loaded) loadFolder();
}

async function loadFolder() {
  folderState.loaded = true;
  folderEls.loading.classList.remove("hidden");
  try {
    const response = await fetch("/v1/accounts/folder");
    const body = await response.json().catch(() => null);
    if (!response.ok) throw new Error(body?.detail || `Could not read the folder (HTTP ${response.status}).`);
    folderState.summary = body;
    folderEls.path.textContent = body.folder;
    folderEls.total.textContent = `${body.total_files} (${body.review_months} months each expected)`;
    for (const [choice, count] of Object.entries(body.counts)) {
      document.querySelector(`#count-${choice}`).textContent = `(${count})`;
    }
    const codes = Object.entries(body.citizen_codes || {});
    folderEls.citizenCodes.textContent = (codes.length
      ? `Codes found: ${codes.map(([code, count]) => `${code} ${count}`).join(" · ")}. Malaysian = ${body.malaysian_code}.`
      : "No citizenship code was found for these accounts in customer_info.")
      + (body.no_profile ? ` ${body.no_profile} account(s) have no customer profile.` : "");
    if (body.unreadable.length) {
      folderEls.unreadable.textContent = `${body.unreadable.length} file(s) could not be read and are left out: `
        + body.unreadable.slice(0, 5).map((item) => `${item.file} (${item.reason})`).join("; ")
        + (body.unreadable.length > 5 ? "; …" : "");
      folderEls.unreadable.classList.remove("hidden");
    }
    folderEls.panel.classList.remove("hidden");
    updateRange(true);
  } catch (error) {
    folderState.loaded = false; // let staff try again by switching back
    showFolderError(error.message || "Could not read the folder.");
  } finally {
    folderEls.loading.classList.add("hidden");
  }
}

function chosenCitizenship() {
  return folderEls.folderForm.querySelector("input[name=citizenship]:checked").value;
}

// Keep the range within the matching files; a new citizenship choice starts again at 1 to 5.
function updateRange(reset) {
  const available = folderState.summary?.counts[chosenCitizenship()] ?? 0;
  for (const input of [folderEls.start, folderEls.end]) input.max = String(Math.max(available, 1));
  if (reset) {
    folderEls.start.value = "1";
    folderEls.end.value = String(Math.min(DEFAULT_RANGE_SIZE, available));
  }
  const { start, end } = chosenRange();
  const valid = Number.isInteger(start) && Number.isInteger(end) && start >= 1 && start <= end && end <= available;
  folderEls.submit.disabled = !valid;
  if (!available) {
    folderEls.countHint.textContent = "No account in the folder matches this choice.";
  } else if (!valid) {
    folderEls.countHint.textContent = `Enter a range within 1 to ${available}, with "From" no bigger than "To".`;
  } else {
    const count = end - start + 1;
    folderEls.countHint.textContent = `Matching files ${start} to ${end} of ${available}, by file name: `
      + `${count} account${count === 1 ? "" : "s"}, about ${duration(count * SECONDS_PER_ACCOUNT)} at one account at a time.`;
  }
  clearFolderError();
}

function chosenRange() {
  return {
    start: Number.parseInt(folderEls.start.value, 10),
    end: Number.parseInt(folderEls.end.value, 10),
  };
}

function duration(seconds) {
  const hours = Math.floor(seconds / 3600);
  const minutes = Math.round((seconds % 3600) / 60);
  return hours ? `${hours} h ${minutes} min` : `${Math.max(minutes, 1)} min`;
}

async function startFolderBatch(event) {
  event.preventDefault();
  const { start, end } = chosenRange();
  folderEls.submit.disabled = true;
  folderEls.submitLabel.textContent = "Queuing…";
  try {
    const response = await fetch("/v1/accounts/folder-batches", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ citizenship: chosenCitizenship(), start, end }),
    });
    const body = await response.json().catch(() => null);
    if (!response.ok) throw new Error(typeof body?.detail === "string" ? body.detail : `Could not start (HTTP ${response.status}).`);
    window.location.href = `/results?batch=${encodeURIComponent(body.batch_id)}`;
  } catch (error) {
    showFolderError(error.message || "Could not start the analysis.");
    folderEls.submit.disabled = false;
    folderEls.submitLabel.textContent = "Start analysis";
  }
}

function showFolderError(message) {
  folderEls.error.textContent = message;
  folderEls.error.classList.remove("hidden");
}

function clearFolderError() {
  folderEls.error.classList.add("hidden");
  folderEls.error.textContent = "";
}
