"""Runs uploaded CSV files one account at a time, in upload order.

Several files can be uploaded together, but they never reach the LLM together:
files wait here, in the application, and one worker analyses one account at a
time. Each account's calls still go through the shared LLM queue, so the number
of calls in flight to the loader can never exceed LLM_CONCURRENCY.

Batches live in memory only (UAT): they are lost when the application restarts,
while the saved HTML/JSON reports stay on disk.
"""
from __future__ import annotations

import asyncio
import logging
from collections import OrderedDict
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal
from uuid import uuid4

logger = logging.getLogger("app.batch_runner")

FileStatus = Literal["queued", "analysing", "done", "failed"]
Analyse = Callable[[str, bytes], Awaitable[dict[str, Any]]]


class FileAnalysisError(Exception):
    """A failure whose message is safe to show to staff, e.g. "CSV is missing column(s): ..."."""


@dataclass
class BatchFile:
    filename: str
    content: bytes | None
    status: FileStatus = "queued"
    result: dict[str, Any] | None = None
    error: str | None = None

    def fail(self, message: str) -> None:
        self.status, self.error, self.content = "failed", message, None


@dataclass
class Batch:
    batch_id: str
    files: list[BatchFile]
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def as_dict(self) -> dict[str, Any]:
        counts = {status: 0 for status in ("queued", "analysing", "done", "failed")}
        for item in self.files:
            counts[item.status] += 1
        return {
            "batch_id": self.batch_id,
            "created_at": self.created_at.isoformat(),
            "total": len(self.files),
            "counts": counts,
            "finished": counts["queued"] == counts["analysing"] == 0,
            "files": [
                {"position": index, "filename": item.filename, "status": item.status,
                 "error": item.error, "result": item.result}
                for index, item in enumerate(self.files, start=1)
            ],
        }


class BatchRunner:
    """One worker, so accounts are analysed strictly one after another across all batches."""

    def __init__(self, analyse: Analyse, *, max_kept_batches: int) -> None:
        if max_kept_batches < 1:
            raise ValueError("max_kept_batches must be at least 1")
        self._analyse = analyse
        self._max_kept = max_kept_batches
        self._batches: OrderedDict[str, Batch] = OrderedDict()
        self._queue: asyncio.Queue[Batch] = asyncio.Queue()
        self._worker: asyncio.Task[None] | None = None

    async def start(self) -> None:
        if self._worker is None:
            self._worker = asyncio.create_task(self._run(), name="batch-runner")
            logger.info("Batch runner started: one account at a time")

    async def close(self) -> None:
        if self._worker is not None:
            self._worker.cancel()
            await asyncio.gather(self._worker, return_exceptions=True)
            self._worker = None
            logger.info("Batch runner stopped")

    def submit(self, files: list[BatchFile]) -> Batch:
        batch = Batch(batch_id=uuid4().hex, files=files)
        self._batches[batch.batch_id] = batch
        while len(self._batches) > self._max_kept:
            self._batches.popitem(last=False)
        self._queue.put_nowait(batch)
        logger.info("Batch queued: batch=%s files=%d batches_waiting=%d",
                    batch.batch_id, len(files), self._queue.qsize())
        return batch

    def get(self, batch_id: str) -> Batch | None:
        return self._batches.get(batch_id)

    async def _run(self) -> None:
        while True:
            batch = await self._queue.get()
            try:
                for position, item in enumerate(batch.files, start=1):
                    if item.status == "queued":
                        await self._analyse_file(batch, position, item)
            finally:
                self._queue.task_done()

    async def _analyse_file(self, batch: Batch, position: int, item: BatchFile) -> None:
        item.status = "analysing"
        logger.info("Batch file started: batch=%s file=%d/%d name=%s",
                    batch.batch_id, position, len(batch.files), item.filename)
        try:
            item.result = await self._analyse(item.filename, item.content or b"")
        except FileAnalysisError as exc:
            item.fail(str(exc))
        except Exception:  # one bad file must not stop the rest of the batch
            logger.exception("Batch file failed unexpectedly: batch=%s name=%s", batch.batch_id, item.filename)
            item.fail("An unexpected error occurred while analysing this file.")
        else:
            item.status, item.content = "done", None
        logger.info("Batch file finished: batch=%s file=%d/%d name=%s status=%s",
                    batch.batch_id, position, len(batch.files), item.filename, item.status)
