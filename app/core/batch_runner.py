"""Runs batches of accounts, a few accounts at a time, started in order.

A batch is either uploaded CSV files or accounts read from the server folder. Files wait
here, in the application; BATCH_ACCOUNT_CONCURRENCY workers each analyse one account at a
time, taking the next file in order when they finish. Running several accounts at once keeps
all the LLM pods busy: one account alone leaves the others idle, above all during its summary
call. Each account's calls still go through the shared LLM queue, so the number of calls in
flight to the loader can never exceed LLM_CONCURRENCY (set it to accounts x 3).

A folder file is read only when its turn comes, so a batch of 400 accounts holds no file
contents in memory. The status poll returns statuses only; each finished report is fetched
on its own (see Batch.as_dict and Batch.file_result), so polling stays small.

Batches live in memory only (UAT): they are lost when the application restarts, while the
saved HTML/JSON reports and the batch results CSV stay on disk.
"""
from __future__ import annotations

import asyncio
import logging
from collections import OrderedDict
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

logger = logging.getLogger("app.batch_runner")

FileStatus = Literal["queued", "analysing", "done", "failed"]
BatchSource = Literal["upload", "folder"]


class FileAnalysisError(Exception):
    """A failure whose message is safe to show to staff, e.g. "CSV is missing column(s): ..."."""


@dataclass
class BatchFile:
    filename: str
    # An uploaded CSV's bytes, or the path of a folder Parquet file (read when its turn comes).
    content: bytes | None = None
    path: Path | None = None
    # Known before the analysis for folder accounts; written to the batch results CSV.
    acct_num: str | None = None
    citizen_cd: str | None = None
    # A folder account's place among the files matching the citizenship choice, e.g. 6 for "start 6".
    list_position: int | None = None
    status: FileStatus = "queued"
    result: dict[str, Any] | None = None
    error: str | None = None

    def fail(self, message: str) -> None:
        self.status, self.error, self.content = "failed", message, None


Analyse = Callable[[BatchFile], Awaitable[dict[str, Any]]]
FileDone = Callable[["Batch", int, BatchFile], Awaitable[None]]


@dataclass
class Batch:
    batch_id: str
    files: list[BatchFile]
    source: BatchSource = "upload"
    # The rule-based risk of every account in a folder batch, for the comparison CSV only.
    expected_risk: str | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    results_csv: str | None = None
    # How many accounts the runner analyses at once, for the results page's wording.
    accounts_at_once: int = 1

    def as_dict(self) -> dict[str, Any]:
        """Statuses only: a report is fetched with file_result, once, when its page is shown."""
        counts = {status: 0 for status in ("queued", "analysing", "done", "failed")}
        for item in self.files:
            counts[item.status] += 1
        return {
            "batch_id": self.batch_id,
            "source": self.source,
            "created_at": self.created_at.isoformat(),
            "total": len(self.files),
            "counts": counts,
            "finished": counts["queued"] == counts["analysing"] == 0,
            "results_csv": self.results_csv,
            "accounts_at_once": self.accounts_at_once,
            "files": [
                {"position": index, "filename": item.filename, "status": item.status, "error": item.error}
                for index, item in enumerate(self.files, start=1)
            ],
        }

    def file_result(self, position: int) -> dict[str, Any] | None:
        """The finished report of the file at ``position`` (1-based); None when not ready."""
        if not 1 <= position <= len(self.files):
            return None
        return self.files[position - 1].result


@dataclass(frozen=True)
class _Job:
    batch: Batch
    position: int
    item: BatchFile


class BatchRunner:
    """``account_workers`` workers share one queue of files, so accounts start in upload/list order
    and at most ``account_workers`` are analysed at once, across all batches."""

    def __init__(
        self, analyse: Analyse, *, max_kept_batches: int, on_file_done: FileDone | None = None,
        account_workers: int = 1,
    ) -> None:
        if max_kept_batches < 1:
            raise ValueError("max_kept_batches must be at least 1")
        if account_workers < 1:
            raise ValueError("account_workers must be at least 1")
        self._analyse = analyse
        self._on_file_done = on_file_done
        self._max_kept = max_kept_batches
        self._account_workers = account_workers
        self._batches: OrderedDict[str, Batch] = OrderedDict()
        self._queue: asyncio.Queue[_Job] = asyncio.Queue()
        self._workers: list[asyncio.Task[None]] = []

    async def start(self) -> None:
        if not self._workers:
            self._workers = [asyncio.create_task(self._run(number), name=f"batch-worker-{number}")
                             for number in range(1, self._account_workers + 1)]
            logger.info("Batch runner started: %d account(s) at a time", self._account_workers)

    async def close(self) -> None:
        for worker in self._workers:
            worker.cancel()
        await asyncio.gather(*self._workers, return_exceptions=True)
        if self._workers:
            logger.info("Batch runner stopped")
        self._workers = []

    def submit(self, files: list[BatchFile], *, source: BatchSource = "upload",
               expected_risk: str | None = None) -> Batch:
        batch = Batch(batch_id=uuid4().hex, files=files, source=source, expected_risk=expected_risk or None,
                      accounts_at_once=self._account_workers)
        self._batches[batch.batch_id] = batch
        while len(self._batches) > self._max_kept:
            self._batches.popitem(last=False)
        for position, item in enumerate(files, start=1):
            self._queue.put_nowait(_Job(batch, position, item))
        logger.info("Batch queued: batch=%s source=%s files=%d files_waiting=%d",
                    batch.batch_id, source, len(files), self._queue.qsize())
        return batch

    def get(self, batch_id: str) -> Batch | None:
        return self._batches.get(batch_id)

    async def _run(self, worker: int) -> None:
        while True:
            job = await self._queue.get()
            try:
                if job.item.status == "queued":
                    await self._analyse_file(worker, job)
                elif job.item.status == "failed":  # rejected at upload: still recorded in the results CSV
                    await self._file_done(job)
            finally:
                self._queue.task_done()

    async def _analyse_file(self, worker: int, job: _Job) -> None:
        batch, position, item = job.batch, job.position, job.item
        item.status = "analysing"
        logger.info("Batch file started: batch=%s worker=%d file=%d/%d name=%s",
                    batch.batch_id, worker, position, len(batch.files), item.filename)
        try:
            item.result = await self._analyse(item)
        except FileAnalysisError as exc:
            item.fail(str(exc))
        except Exception:  # one bad file must not stop the rest of the batch
            logger.exception("Batch file failed unexpectedly: batch=%s name=%s", batch.batch_id, item.filename)
            item.fail("An unexpected error occurred while analysing this file.")
        else:
            item.status, item.content = "done", None
        logger.info("Batch file finished: batch=%s worker=%d file=%d/%d name=%s status=%s",
                    batch.batch_id, worker, position, len(batch.files), item.filename, item.status)
        await self._file_done(job)

    async def _file_done(self, job: _Job) -> None:
        if self._on_file_done is None:
            return
        try:
            await self._on_file_done(job.batch, job.position, job.item)
        except Exception:  # the results CSV must never stop the analysis
            logger.exception("Could not record a finished file: batch=%s name=%s",
                             job.batch.batch_id, job.item.filename)
