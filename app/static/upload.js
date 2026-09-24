// Upload page — file selection and hand-off to the results page.
//
// The analysis result is passed to /results via sessionStorage rather than a
// server-rendered page, since this app serves plain static HTML with no
// templating layer. This keeps the two pages independent: this file only
// ever deals with "get a valid CSV to the API", nothing about rendering
// findings.

const MAX_UPLOAD_BYTES = 2 * 1024 * 1024;
const RESULT_STORAGE_KEY = "dd:last-result";
const DEFAULT_FILE_HINT = "Drop CSV file here";

const els = {
  form: document.querySelector("#analysis-form"),
  fileInput: document.querySelector("#file"),
  fileField: document.querySelector("#file-field"),
  fileName: document.querySelector("#file-name"),
  submitButton: document.querySelector("#submit-button"),
  submitButtonLabel: document.querySelector("#submit-button-label"),
  loading: document.querySelector("#loading"),
  error: document.querySelector("#error"),
};

init();

function init() {
  els.fileInput.addEventListener("change", () => {
    handleFileSelected(els.fileInput.files[0]);
  });

  ["dragenter", "dragover"].forEach((eventName) => {
    els.fileField.addEventListener(eventName, (event) => {
      event.preventDefault();
      els.fileField.classList.add("is-dragging");
    });
  });

  ["dragleave", "drop"].forEach((eventName) => {
    els.fileField.addEventListener(eventName, (event) => {
      event.preventDefault();
      els.fileField.classList.remove("is-dragging");
    });
  });

  els.fileField.addEventListener("drop", (event) => {
    const file = event.dataTransfer?.files?.[0];
    if (file) {
      els.fileInput.files = event.dataTransfer.files;
      handleFileSelected(file);
    }
  });

  els.form.addEventListener("submit", handleSubmit);
}

function handleFileSelected(file) {
  if (!file) {
    els.fileName.textContent = DEFAULT_FILE_HINT;
    return;
  }

  const isCsv = file.name.toLowerCase().endsWith(".csv");
  if (!isCsv) {
    showError("Please choose a .csv file exported from online banking or a core system.");
    els.fileInput.value = "";
    els.fileName.textContent = DEFAULT_FILE_HINT;
    return;
  }

  if (file.size > MAX_UPLOAD_BYTES) {
    showError(`"${file.name}" is larger than the 2 MB limit for this workspace.`);
    els.fileInput.value = "";
    els.fileName.textContent = DEFAULT_FILE_HINT;
    return;
  }

  clearError();
  els.fileName.textContent = file.name;
}

async function handleSubmit(event) {
  event.preventDefault();

  const file = els.fileInput.files[0];
  if (!file) {
    showError("Choose a CSV statement before running an analysis.");
    return;
  }

  const payload = new FormData();
  payload.append("file", file);

  setBusy(true);

  try {
    const response = await fetch("/v1/accounts/analyze-file", {
      method: "POST",
      body: payload,
    });

    const body = await response.json().catch(() => null);

    if (!response.ok) {
      throw new Error(body?.detail || `Statement analysis failed (HTTP ${response.status}).`);
    }

    sessionStorage.setItem(RESULT_STORAGE_KEY, JSON.stringify(body));
    window.location.href = "/results";
  } catch (error) {
    showError(error.message || "Statement analysis failed. Please try again.");
    setBusy(false);
  }
}

function setBusy(isBusy) {
  els.submitButton.disabled = isBusy;
  els.submitButtonLabel.textContent = isBusy ? "Analysing\u2026" : "Start analysis";
  els.loading.classList.toggle("hidden", !isBusy);
  if (isBusy) clearError();
}

function showError(message) {
  els.error.textContent = message;
  els.error.classList.remove("hidden");
}

function clearError() {
  els.error.classList.add("hidden");
  els.error.textContent = "";
}
