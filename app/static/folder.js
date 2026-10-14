// Upload page — "Read from folder": analyse accounts straight from the server folder.
//
// The folder itself is fixed on the server (ACCOUNT_FOLDER_PATH); staff choose the
// citizenship and how many accounts. The first N matching files (by file name) are
// queued as one batch and followed on the same results page as an upload.

const SECONDS_PER_ACCOUNT = 80; // rough guide from UAT runs, for the time estimate only

const folderEls = {
  switchButtons: document.querySelectorAll(".source-option"),
  uploadForm: document.querySelector("#analysis-form"),
  folderForm: document.querySelector("#folder-form"),
  loading: document.querySelector("#folder-loading"),
  panel: document.querySelector("#folder-panel"),
  path: document.querySelector("#folder-path"),
  total: document.querySelector("#folder-total"),
  count: document.querySelector("#account-count"),
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
  if (event.target.name === "citizenship") updateCount(true);
});
folderEls.count.addEventListener("input", () => updateCount(false));
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
    updateCount(true);
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

// Keep the number within what matches; a new citizenship choice starts at all of its accounts.
function updateCount(reset) {
  const available = folderState.summary?.counts[chosenCitizenship()] ?? 0;
  folderEls.count.max = String(Math.max(available, 1));
  if (reset) folderEls.count.value = String(available);
  const requested = Number.parseInt(folderEls.count.value, 10);
  const valid = Number.isInteger(requested) && requested >= 1 && requested <= available;
  folderEls.submit.disabled = !valid;
  if (!available) {
    folderEls.countHint.textContent = "No account in the folder matches this choice.";
  } else if (!valid) {
    folderEls.countHint.textContent = `Enter a number from 1 to ${available}.`;
  } else {
    folderEls.countHint.textContent = `The first ${requested} of ${available} matching files, by file name. `
      + `About ${duration(requested * SECONDS_PER_ACCOUNT)} at one account at a time.`;
  }
  clearFolderError();
}

function duration(seconds) {
  const hours = Math.floor(seconds / 3600);
  const minutes = Math.round((seconds % 3600) / 60);
  return hours ? `${hours} h ${minutes} min` : `${Math.max(minutes, 1)} min`;
}

async function startFolderBatch(event) {
  event.preventDefault();
  const count = Number.parseInt(folderEls.count.value, 10);
  folderEls.submit.disabled = true;
  folderEls.submitLabel.textContent = "Queuing…";
  try {
    const response = await fetch("/v1/accounts/folder-batches", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ citizenship: chosenCitizenship(), count }),
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
