import asyncio
import logging
import time
from collections.abc import Callable
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from app.adapters.llm_client import LlmContextWindowError, LlmServiceError
from app.config import Settings
from app.core.analysis_service import AnalysisService, ModelOutputError
from app.core.account_folder import AccountFolderError, CitizenshipChoice, scan_folder
from app.core.batch_results import BatchResultsLog
from app.core.batch_runner import Batch, BatchFile, BatchRunner, BatchSource, FileAnalysisError
from app.core.llm_work_queue import LlmQueueClosedError, LlmQueueFullError
from app.core.models import AccountAnalysisRequest, MonthlySummaryRow
from app.core.report_store import ReportStore
from app.core.summary_loader import (
    CustomerProfileError,
    SummaryCsvError,
    generate_case_id,
    load_customer_profile,
    parse_monthly_summary_csv,
    read_monthly_summary_parquet,
)

logger = logging.getLogger("app.api")

# A 12-row monthly summary is tiny; this only guards against the wrong file
# being dropped on the endpoint before it reaches the parser.
_MAX_UPLOAD_BYTES = 2 * 1024 * 1024


class UploadAnalyser:
    """Analyses one account, from an uploaded CSV or a folder Parquet file. Used by the
    single-file endpoint and by the batch runner, so an account gives the same result
    whichever way it arrived."""

    def __init__(
        self, service: AnalysisService, report_store: ReportStore, customer_info_path: Path, review_months: int,
    ) -> None:
        self._service = service
        self._report_store = report_store
        self._customer_info_path = customer_info_path
        self._review_months = review_months

    async def analyse(self, filename: str, raw: bytes) -> dict:
        """One uploaded CSV: the assessment and report links; raises HTTPException with a message for staff."""
        if problem := upload_problem(filename, len(raw)):
            raise HTTPException(status_code=problem[0], detail=problem[1])
        return await self._analyse_rows(
            filename, lambda: parse_monthly_summary_csv(raw.decode("utf-8-sig"), self._review_months))

    async def analyse_folder_file(self, path: Path) -> dict:
        """One Parquet file from the account folder, read now (in a thread: Parquet I/O is blocking)."""
        return await self._analyse_rows(
            path.name, lambda: read_monthly_summary_parquet(path, self._review_months), in_thread=True)

    async def _analyse_rows(
        self, filename: str, load_rows: Callable[[], list[MonthlySummaryRow]], *, in_thread: bool = False,
    ) -> dict:
        started_at = time.monotonic()
        case_id = "unknown"
        try:
            monthly_summary = await asyncio.to_thread(load_rows) if in_thread else load_rows()
            acct_num = monthly_summary[0].acct_num
            case_id = generate_case_id(acct_num)
            logger.info("Starting analysis: case=%s acct=%s months=%s", case_id, acct_num, len(monthly_summary))

            # Parquet I/O is blocking; keep it off the event loop.
            customer_profile = await asyncio.to_thread(load_customer_profile, self._customer_info_path, acct_num)

            request = AccountAnalysisRequest(
                case_id=case_id,
                source_filename=filename or "monthly-summary.csv",
                acct_num=acct_num,
                monthly_summary=monthly_summary,
                customer_profile=customer_profile,
            )

            result = await self._service.analyze_account(request)
            # Report rendering and disk I/O must not block queue workers or
            # other concurrent uploads on the event loop.
            saved = await asyncio.to_thread(self._report_store.save, result)

            logger.info(
                "Analysis complete: case=%s decision=%s risk=%s duration=%.1fs",
                case_id, result.decision, result.risk_level, time.monotonic() - started_at,
            )
            return {
                **result.model_dump(mode="json"),
                "source_filename": filename,
                "report_html": f"/v1/accounts/reports/{saved.html_name}",
                "report_json": f"/v1/accounts/reports/{saved.json_name}",
            }
        except UnicodeDecodeError as exc:
            raise HTTPException(status_code=422, detail="CSV must be saved as UTF-8.") from exc
        except OSError as exc:  # a folder file moved or unreadable since the scan
            raise HTTPException(status_code=422, detail=f"Could not read {filename}: {exc}") from exc
        except SummaryCsvError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except CustomerProfileError as exc:
            logger.error("customer_info lookup failed: case=%s error=%s", case_id, exc)
            raise HTTPException(
                status_code=502,
                detail="Could not read the shared customer profile data. Please contact support.",
            ) from exc
        except ModelOutputError as exc:
            logger.error("Model output did not match schema: case=%s error=%s", case_id, exc)
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        except LlmContextWindowError as exc:
            logger.error(
                "LLM context-window error: case=%s upstream_status=%s upstream_request_id=%s error=%s",
                case_id, exc.status_code, exc.upstream_request_id, exc,
            )
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except LlmServiceError as exc:
            logger.error("LLM service call failed: case=%s error=%s", case_id, exc)
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        except LlmQueueFullError as exc:
            logger.warning("LLM queue capacity reached: case=%s", case_id)
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except LlmQueueClosedError as exc:
            logger.warning("LLM queue unavailable during shutdown: case=%s", case_id)
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    async def analyse_for_batch(self, item: BatchFile) -> dict:
        """The same analysis, with the failure message the batch shows for that file."""
        try:
            if item.path is not None:
                return await self.analyse_folder_file(item.path)
            return await self.analyse(item.filename, item.content or b"")
        except HTTPException as exc:
            raise FileAnalysisError(str(exc.detail)) from exc


def upload_problem(filename: str | None, size: int) -> tuple[int, str] | None:
    """(HTTP status, message) when the upload cannot be a monthly-summary CSV; None when it may be."""
    if not (filename or "").lower().endswith(".csv"):
        return 415, "Please upload the monthly summary CSV for one account."
    if size > _MAX_UPLOAD_BYTES:
        return 413, "The CSV exceeds the 2 MB upload limit for a monthly summary."
    return None


class FolderBatchRequest(BaseModel):
    citizenship: CitizenshipChoice
    count: int = Field(ge=1)


def build_router(
    analyser: UploadAnalyser,
    batch_runner: BatchRunner,
    report_store: ReportStore,
    results_log: BatchResultsLog,
    settings: Settings,
) -> APIRouter:
    router = APIRouter(prefix="/v1/accounts", tags=["accounts"])

    def submit(files: list[BatchFile], source: BatchSource, expected_risk: str | None = None) -> Batch:
        # No await between submit and start: the header is written before the worker runs.
        batch = batch_runner.submit(files, source=source, expected_risk=expected_risk)
        results_log.start(batch)
        return batch

    async def scan():
        folder = settings.account_folder
        if folder is None:
            raise HTTPException(status_code=404, detail="Reading from a folder is not set up (ACCOUNT_FOLDER_PATH).")
        try:
            return await asyncio.to_thread(scan_folder, folder, settings.customer_info_path)
        except AccountFolderError as exc:
            logger.error("Account folder scan failed: %s", exc)
            raise HTTPException(status_code=502, detail=str(exc)) from exc

    @router.get("/folder")
    async def folder_summary() -> dict:
        """What the folder holds: files, and how many accounts match each citizenship choice."""
        return {**(await scan()).summary(settings.malaysian_citizen_cd), "review_months": settings.review_months,
                "malaysian_code": settings.malaysian_citizen_cd}

    @router.post("/folder-batches", status_code=202)
    async def create_folder_batch(request: FolderBatchRequest) -> dict:
        """The first ``count`` accounts (by file name) matching the citizenship choice, analysed
        one at a time like an upload. Poll GET /v1/accounts/batches/{batch_id}."""
        accounts = (await scan()).matching(request.citizenship, settings.malaysian_citizen_cd)
        if not accounts:
            raise HTTPException(status_code=422, detail="No account in the folder matches this citizenship choice.")
        chosen = accounts[:request.count]
        logger.info("Folder batch: citizenship=%s requested=%d matching=%d chosen=%d",
                    request.citizenship, request.count, len(accounts), len(chosen))
        files = [BatchFile(filename=account.path.name, path=account.path, acct_num=account.acct_num,
                           citizen_cd=account.citizen_cd) for account in chosen]
        return submit(files, "folder", settings.folder_expected_risk).as_dict()

    @router.post("/analyze-file")
    async def analyze_file(file: UploadFile = File(...)) -> dict:
        """One CSV, analysed now; the response is the assessment."""
        return await analyser.analyse(file.filename or "", await file.read())

    @router.post("/batches", status_code=202)
    async def create_batch(files: list[UploadFile] = File(...)) -> dict:
        """Several CSVs, one account each. They are queued and analysed one at a time;
        poll GET /v1/accounts/batches/{batch_id} for each file's status and result."""
        if not files:
            raise HTTPException(status_code=422, detail="Choose at least one CSV file.")
        if len(files) > settings.batch_max_files:
            raise HTTPException(
                status_code=413,
                detail=f"Upload at most {settings.batch_max_files} CSV files at a time; {len(files)} were selected.",
            )
        items = []
        for upload in files:
            raw = await upload.read()
            item = BatchFile(filename=upload.filename or "unnamed.csv", content=raw)
            if problem := upload_problem(upload.filename, len(raw)):
                item.fail(problem[1])  # a wrong file is reported, the others still run
            items.append(item)
        return submit(items, "upload").as_dict()

    def find_batch(batch_id: str) -> Batch:
        batch = batch_runner.get(batch_id)
        if batch is None:
            raise HTTPException(
                status_code=404,
                detail="Batch not found. Batches are kept in memory and are cleared when the application restarts.",
            )
        return batch

    @router.get("/batches/{batch_id}")
    def batch_status(batch_id: str) -> dict:
        """Each file's status; fetch a finished report with /batches/{batch_id}/files/{position}."""
        return find_batch(batch_id).as_dict()

    @router.get("/batches/{batch_id}/files/{position}")
    def batch_file_result(batch_id: str, position: int) -> dict:
        result = find_batch(batch_id).file_result(position)
        if result is None:
            raise HTTPException(status_code=404, detail="This report is not ready.")
        return result

    @router.get("/batches/{batch_id}/results.csv")
    def batch_results_csv(batch_id: str) -> FileResponse:
        batch = find_batch(batch_id)
        path = results_log.resolve(batch.results_csv or "")
        if path is None:
            raise HTTPException(status_code=404, detail="The results CSV for this batch was not found.")
        return FileResponse(path, media_type="text/csv", filename=path.name)

    @router.get("/reports/{report_name}")
    def download_report(report_name: str) -> FileResponse:
        report_path = report_store.resolve(report_name)
        if report_path is None:
            raise HTTPException(status_code=404, detail="Report not found.")
        media_type = "application/json" if report_path.suffix == ".json" else "text/html"
        return FileResponse(report_path, media_type=media_type, filename=report_path.name)

    return router
