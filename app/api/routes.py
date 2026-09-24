import asyncio
import logging
import time
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import FileResponse

from app.adapters.llm_client import LlmContextWindowError, LlmServiceError
from app.core.analysis_service import AnalysisService, ModelOutputError
from app.core.llm_work_queue import LlmQueueClosedError, LlmQueueFullError
from app.core.models import AccountAnalysisRequest
from app.core.report_store import ReportStore
from app.core.summary_loader import (
    CustomerProfileError,
    SummaryCsvError,
    generate_case_id,
    load_customer_profile,
    parse_monthly_summary_csv,
    review_window,
)

logger = logging.getLogger("app.api")

# A 6-row monthly summary is tiny; this only guards against the wrong file
# being dropped on the endpoint before it reaches the parser.
_MAX_UPLOAD_BYTES = 2 * 1024 * 1024


def build_router(
    service: AnalysisService,
    report_store: ReportStore,
    customer_info_path: Path,
) -> APIRouter:
    router = APIRouter(prefix="/v1/accounts", tags=["accounts"])

    @router.post("/analyze-file")
    async def analyze_file(file: UploadFile = File(...)) -> dict:
        if not (file.filename or "").lower().endswith(".csv"):
            raise HTTPException(
                status_code=415,
                detail="Please upload the 6-month monthly summary CSV for one account.",
            )

        raw = await file.read()
        if len(raw) > _MAX_UPLOAD_BYTES:
            raise HTTPException(
                status_code=413,
                detail="The CSV exceeds the 2 MB upload limit for a 6-row monthly summary.",
            )

        started_at = time.monotonic()
        case_id = "unknown"
        try:
            content = raw.decode("utf-8-sig")
            monthly_summary = parse_monthly_summary_csv(content)
            acct_num = monthly_summary[0].acct_num
            case_id = generate_case_id(acct_num)
            logger.info("Starting analysis: case=%s acct=%s months=%s", case_id, acct_num, len(monthly_summary))

            window_start, window_end = review_window(monthly_summary)
            # Parquet I/O is blocking; keep it off the event loop.
            customer_profile = await asyncio.to_thread(
                load_customer_profile, customer_info_path, acct_num, window_start, window_end
            )

            request = AccountAnalysisRequest(
                case_id=case_id,
                source_filename=file.filename or "monthly-summary.csv",
                acct_num=acct_num,
                monthly_summary=monthly_summary,
                customer_profile=customer_profile,
            )

            result = await service.analyze_account(request)
            # Report rendering and disk I/O must not block queue workers or
            # other concurrent uploads on the event loop.
            saved = await asyncio.to_thread(report_store.save, result)

            logger.info(
                "Analysis complete: case=%s decision=%s risk=%s duration=%.1fs",
                case_id, result.decision, result.risk_level, time.monotonic() - started_at,
            )
            return {
                **result.model_dump(mode="json"),
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

    @router.get("/reports/{report_name}")
    def download_report(report_name: str) -> FileResponse:
        report_path = report_store.resolve(report_name)
        if report_path is None:
            raise HTTPException(status_code=404, detail="Report not found.")
        media_type = "application/json" if report_path.suffix == ".json" else "text/html"
        return FileResponse(report_path, media_type=media_type, filename=report_path.name)

    return router