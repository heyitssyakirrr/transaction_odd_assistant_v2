// Upload page — choose one or more CSV files (one account each) and queue them.
//
// The files are sent together to /v1/accounts/batches. The server queues them and
// analyses one account at a time; the results page (/results?batch=...) follows
// their progress and shows one report per page.

const MAX_UPLOAD_BYTES = 2 * 1024 * 1024;
const MAX_FILES = 20; // keep in line with BATCH_MAX_FILES on the server
const DEFAULT_FILE_HINT = "Drop CSV files here";

const els = {
  form: document.querySelector("#analysis-form"),
  fileInput: document.querySelector("#file"),
  fileField: document.querySelector("#file-field"),
  fileName: document.querySelector("#file-name"),
  fileList: document.querySelector("#file-list"),
  submitButton: document.querySelector("#submit-button"),
  submitButtonLabel: document.querySelector("#submit-button-label"),
  loading: document.querySelector("#loading"),
  error: document.querySelector("#error"),
};

init();

function init() {
  els.fileInput.addEventListener("change", () => handleFilesSelected(els.fileInput.files));

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
    const files = event.dataTransfer?.files;
    if (files && files.length) {
      els.fileInput.files = files;
      handleFilesSelected(files);
    }
  });

  els.form.addEventListener("submit", handleSubmit);
}

// Every problem with the chosen files, as messages for staff (empty when all are fine).
function fileProblems(files) {
  const problems = [];
  if (files.length > MAX_FILES) {
    problems.push(`Choose at most ${MAX_FILES} files at a time; ${files.length} were selected.`);
  }
  for (const file of files) {
    if (!file.name.toLowerCase().endsWith(".csv")) {
      problems.push(`"${file.name}" is not a .csv file.`);
    } else if (file.size > MAX_UPLOAD_BYTES) {
      problems.push(`"${file.name}" is larger than the 2 MB limit.`);
    }
  }
  return problems;
}

function handleFilesSelected(fileList) {
  const files = Array.from(fileList || []);
  if (!files.length) {
    resetSelection();
    return;
  }

  const problems = fileProblems(files);
  if (problems.length) {
    showError(problems.join(" "));
    els.fileInput.value = "";
    resetSelection();
    return;
  }

  clearError();
  els.fileName.textContent = files.length === 1 ? files[0].name : `${files.length} files selected`;
  els.fileList.innerHTML = files.map((file) => `<li>${escapeHtml(file.name)}</li>`).join("");
  els.fileList.classList.toggle("hidden", files.length < 2);
}

function resetSelection() {
  els.fileName.textContent = DEFAULT_FILE_HINT;
  els.fileList.innerHTML = "";
  els.fileList.classList.add("hidden");
}

async function handleSubmit(event) {
  event.preventDefault();

  const files = Array.from(els.fileInput.files || []);
  if (!files.length) {
    showError("Choose at least one CSV file before running an analysis.");
    return;
  }
  const problems = fileProblems(files);
  if (problems.length) {
    showError(problems.join(" "));
    return;
  }

  const payload = new FormData();
  files.forEach((file) => payload.append("files", file));

  setBusy(true);

  try {
    const response = await fetch("/v1/accounts/batches", { method: "POST", body: payload });
    const body = await response.json().catch(() => null);

    if (!response.ok) {
      throw new Error(body?.detail || `Upload failed (HTTP ${response.status}).`);
    }

    window.location.href = `/results?batch=${encodeURIComponent(body.batch_id)}`;
  } catch (error) {
    showError(error.message || "Upload failed. Please try again.");
    setBusy(false);
  }
}

function setBusy(isBusy) {
  els.submitButton.disabled = isBusy;
  els.submitButtonLabel.textContent = isBusy ? "Uploading…" : "Start analysis";
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

function escapeHtml(value) {
  const element = document.createElement("div");
  element.textContent = value ?? "";
  return element.innerHTML;
}
