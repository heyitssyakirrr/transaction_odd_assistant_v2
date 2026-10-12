import asyncio
import logging
import time
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import FileResponse

from app.adapters.llm_client import LlmContextWindowError, LlmServiceError
from app.core.analysis_service import AnalysisService, ModelOutputError
from app.core.batch_runner import BatchFile, BatchRunner, FileAnalysisError
from app.core.llm_work_queue import LlmQueueClosedError, LlmQueueFullError
from app.core.models import AccountAnalysisRequest
from app.core.report_store import ReportStore
from app.core.summary_loader import (
    CustomerProfileError,
    SummaryCsvError,
    generate_case_id,
    load_customer_profile,
    parse_monthly_summary_csv,
)

logger = logging.getLogger("app.api")

# A 6-row monthly summary is tiny; this only guards against the wrong file
# being dropped on the endpoint before it reaches the parser.
_MAX_UPLOAD_BYTES = 2 * 1024 * 1024


class UploadAnalyser:
    """Analyses one uploaded CSV. Used by the single-file endpoint and by the batch runner,
    so a file gives the same result whichever way it was uploaded."""

    def __init__(self, service: AnalysisService, report_store: ReportStore, customer_info_path: Path) -> None:
        self._service = service
        self._report_store = report_store
        self._customer_info_path = customer_info_path

    async def analyse(self, filename: str, raw: bytes) -> dict:
        """The assessment and report links; raises HTTPException with a message for staff."""
        if problem := upload_problem(filename, len(raw)):
            raise HTTPException(status_code=problem[0], detail=problem[1])

        started_at = time.monotonic()
        case_id = "unknown"
        try:
            content = raw.decode("utf-8-sig")
            monthly_summary = parse_monthly_summary_csv(content)
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

    async def analyse_for_batch(self, filename: str, raw: bytes) -> dict:
        """The same analysis, with the failure message the batch shows for that file."""
        try:
            return await self.analyse(filename, raw)
        except HTTPException as exc:
            raise FileAnalysisError(str(exc.detail)) from exc


def upload_problem(filename: str | None, size: int) -> tuple[int, str] | None:
    """(HTTP status, message) when the upload cannot be a monthly-summary CSV; None when it may be."""
    if not (filename or "").lower().endswith(".csv"):
        return 415, "Please upload the 6-month monthly summary CSV for one account."
    if size > _MAX_UPLOAD_BYTES:
        return 413, "The CSV exceeds the 2 MB upload limit for a 6-row monthly summary."
    return None


def build_router(
    analyser: UploadAnalyser,
    batch_runner: BatchRunner,
    report_store: ReportStore,
    batch_max_files: int,
) -> APIRouter:
    router = APIRouter(prefix="/v1/accounts", tags=["accounts"])

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
        if len(files) > batch_max_files:
            raise HTTPException(
                status_code=413,
                detail=f"Upload at most {batch_max_files} CSV files at a time; {len(files)} were selected.",
            )
        items = []
        for upload in files:
            raw = await upload.read()
            item = BatchFile(filename=upload.filename or "unnamed.csv", content=raw)
            if problem := upload_problem(upload.filename, len(raw)):
                item.fail(problem[1])  # a wrong file is reported, the others still run
            items.append(item)
        return batch_runner.submit(items).as_dict()

    @router.get("/batches/{batch_id}")
    def batch_status(batch_id: str) -> dict:
        batch = batch_runner.get(batch_id)
        if batch is None:
            raise HTTPException(
                status_code=404,
                detail="Batch not found. Batches are kept in memory and are cleared when the application restarts.",
            )
        return batch.as_dict()

    @router.get("/reports/{report_name}")
    def download_report(report_name: str) -> FileResponse:
        report_path = report_store.resolve(report_name)
        if report_path is None:
            raise HTTPException(status_code=404, detail="Report not found.")
        media_type = "application/json" if report_path.suffix == ".json" else "text/html"
        return FileResponse(report_path, media_type=media_type, filename=report_path.name)

    return router
