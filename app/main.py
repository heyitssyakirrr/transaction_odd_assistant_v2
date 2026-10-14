import logging
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.adapters.llm_client import OpenAICompatibleClient
from app.api.routes import UploadAnalyser, build_router
from app.config import Settings
from app.core.analysis_service import AnalysisService
from app.core.batch_results import BatchResultsLog
from app.core.batch_runner import BatchRunner
from app.core.llm_work_queue import LlmWorkQueue
from app.core.report_store import ReportStore
from app.logging_config import LOG_FORMAT, configure_persistent_logging

logging.basicConfig(
    level=logging.INFO,
    format=LOG_FORMAT,
)
logger = logging.getLogger("app.main")

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent

settings = Settings()
log_file = configure_persistent_logging(settings.log_directory_path, settings.log_retention_days)

if log_file is not None:
    logger.info("Persistent application logging enabled: %s", log_file)

if not settings.llm_base_url:
    logger.warning(
        "LLM_BASE_URL is not set. Copy .env.example to '.env' in the project "
        "root (next to Dockerfile/README.md, not inside app/) and set your "
        "loader's URL before analysing statements."
    )
else:
    logger.info(
        "LLM loader configured: %s (model=%s, concurrency=%s, accounts_at_once=%s, helper_id=%s)",
        settings.chat_url,
        settings.llm_model,
        settings.llm_concurrency,
        settings.batch_account_concurrency,
        "set" if settings.helper_id else "not set",
    )
if settings.llm_concurrency < settings.llm_calls_needed:
    logger.warning(
        "LLM_CONCURRENCY=%d is below BATCH_ACCOUNT_CONCURRENCY x 3 = %d: accounts will wait in the app "
        "for an LLM slot instead of keeping every pod busy. Set LLM_CONCURRENCY=%d.",
        settings.llm_concurrency, settings.llm_calls_needed, settings.llm_calls_needed,
    )

llm_client = OpenAICompatibleClient(settings)
llm_queue = LlmWorkQueue(
    worker_count=settings.llm_concurrency,
    maxsize=settings.llm_queue_maxsize,
    enqueue_timeout_seconds=settings.llm_queue_enqueue_timeout_seconds,
)
service = AnalysisService(llm_client, settings, llm_queue)

report_store = ReportStore(settings.report_directory_path)
analyser = UploadAnalyser(service, report_store, settings.customer_info_path, settings.review_months)
results_log = BatchResultsLog(settings.batch_results_path)
batch_runner = BatchRunner(
    analyser.analyse_for_batch, max_kept_batches=settings.batch_keep_recent, on_file_done=results_log.record,
    account_workers=settings.batch_account_concurrency,
)

app = FastAPI(
    title="Transaction Due-Diligence Assistant",
    version="0.3.0",
)

app.mount(
    "/static",
    StaticFiles(directory=BASE_DIR / "static"),
    name="static",
)

app.include_router(build_router(analyser, batch_runner, report_store, results_log, settings))


@app.middleware("http")
async def always_check_for_newer_pages(request: Request, call_next):
    """Make browsers re-check pages and scripts on every visit (a cheap 304 when unchanged),
    so an updated results.js is never replaced by a stale cached copy."""
    response = await call_next(request)
    path = request.url.path
    if path in ("/", "/results") or path.startswith("/static/"):
        response.headers["Cache-Control"] = "no-cache"
    return response


@app.get("/", include_in_schema=False)
def workspace() -> FileResponse:
    return FileResponse(BASE_DIR / "templates" / "index.html")


@app.get("/results", include_in_schema=False)
def results_page() -> FileResponse:
    return FileResponse(BASE_DIR / "templates" / "results.html")


@app.get("/healthz")
def healthz() -> dict[str, object]:
    """Liveness probe: confirms this process can answer HTTP requests."""
    return {
        "status": "ok",
        "model": settings.llm_model,
        "llm_configured": bool(settings.llm_base_url),
        "llm_queue": {"queued": llm_queue.queued_jobs, "active": llm_queue.active_jobs, "workers": settings.llm_concurrency},
    }


@app.get("/readyz")
def readyz() -> JSONResponse:
    """Readiness probe: do not route uploads until the queue can accept work."""
    ready = bool(settings.llm_base_url) and llm_queue.is_running
    return JSONResponse(
        status_code=200 if ready else 503,
        content={
            "status": "ready" if ready else "not_ready",
            "llm_configured": bool(settings.llm_base_url),
            "llm_queue_running": llm_queue.is_running,
        },
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled error on %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=500,
        content={"detail": "An unexpected error occurred. Please try again or contact support."},
    )


@app.on_event("startup")
async def start_llm_queue() -> None:
    await llm_queue.start()
    await batch_runner.start()


@app.on_event("shutdown")
async def close_llm_client() -> None:
    await batch_runner.close()
    await llm_queue.close()
    await llm_client.close()


if __name__ == "__main__":
    import uvicorn

    # Do not watch broad warehouse directories while an AML review is running.
    # OpenShift uses the Dockerfile command without reload as well.
    uvicorn.run("app.main:app", host="127.0.0.1", port=5000, reload=False)
