from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from app.core.call_timing import CallMeasure


logger = logging.getLogger("app.llm_queue")


class LlmQueueFullError(RuntimeError):
    """Raised when the application cannot safely accept another LLM job."""


class LlmQueueClosedError(RuntimeError):
    """Raised when an LLM job is submitted while the application is stopping."""


@dataclass
class _WorkItem:
    name: str
    operation: Callable[[], Awaitable[Any]]
    result: asyncio.Future[Any]
    enqueued_at: float
    measure: CallMeasure | None = None


class LlmWorkQueue:
    """A process-local, bounded FIFO queue for every call to the LLM service.

    A worker owns one complete request, including its retry policy.  Therefore
    the number of active HTTP conversations can never exceed ``worker_count``.
    The bounded queue provides backpressure instead of allowing one large CSV
    or several simultaneous uploads to create an unbounded number of tasks.

    This limit is per application process. Deploy one Uvicorn worker for a
    three-replica LLM service, or divide the limit across application replicas.
    """

    _STOP = object()

    def __init__(self, *, worker_count: int, maxsize: int, enqueue_timeout_seconds: float) -> None:
        if worker_count < 1:
            raise ValueError("worker_count must be at least 1")
        if maxsize < 1:
            raise ValueError("maxsize must be at least 1")
        if enqueue_timeout_seconds <= 0:
            raise ValueError("enqueue_timeout_seconds must be greater than 0")

        self._worker_count = worker_count
        self._queue: asyncio.Queue[_WorkItem | object] = asyncio.Queue(maxsize=maxsize)
        self._enqueue_timeout_seconds = enqueue_timeout_seconds
        self._workers: list[asyncio.Task[None]] = []
        self._accepting = False
        self._active = 0

    @property
    def queued_jobs(self) -> int:
        return self._queue.qsize()

    @property
    def active_jobs(self) -> int:
        return self._active

    @property
    def is_running(self) -> bool:
        return self._accepting and bool(self._workers)

    async def start(self) -> None:
        if self._workers:
            return
        self._accepting = True
        self._workers = [
            asyncio.create_task(self._worker(index), name=f"llm-worker-{index}")
            for index in range(1, self._worker_count + 1)
        ]
        logger.info("LLM queue started: workers=%d capacity=%d", self._worker_count, self._queue.maxsize)

    async def submit(
        self, name: str, operation: Callable[[], Awaitable[Any]], measure: CallMeasure | None = None,
    ) -> Any:
        """Run ``operation`` on the next free worker. ``measure``, when given, receives the time
        this job waited in the queue and the time the LLM took once a worker sent it."""
        if not self._accepting:
            raise LlmQueueClosedError("The LLM queue is not accepting new work.")

        loop = asyncio.get_running_loop()
        item = _WorkItem(name, operation, loop.create_future(), time.monotonic(), measure)
        try:
            await asyncio.wait_for(self._queue.put(item), timeout=self._enqueue_timeout_seconds)
        except TimeoutError as exc:
            logger.warning(
                "LLM queue is full; rejecting job=%s queued=%d active=%d capacity=%d",
                name, self.queued_jobs, self.active_jobs, self._queue.maxsize,
            )
            raise LlmQueueFullError("LLM queue is full; retry the analysis shortly.") from exc

        logger.info("LLM job queued: job=%s queued=%d active=%d", name, self.queued_jobs, self.active_jobs)
        try:
            # Do not cancel an already-running LLM request just because the
            # browser disconnected. The worker will finish and release its slot.
            return await asyncio.shield(item.result)
        except asyncio.CancelledError:
            item.result.cancel()
            logger.info("LLM job caller cancelled: job=%s", name)
            raise

    async def close(self) -> None:
        self._accepting = False
        for _ in self._workers:
            await self._queue.put(self._STOP)
        await asyncio.gather(*self._workers, return_exceptions=True)
        self._workers.clear()
        logger.info("LLM queue stopped")

    async def _worker(self, worker_id: int) -> None:
        while True:
            queued_item = await self._queue.get()
            try:
                if queued_item is self._STOP:
                    return
                item = queued_item
                if item.result.cancelled():
                    logger.info("LLM job skipped because caller cancelled: job=%s", item.name)
                    continue

                self._active += 1
                queued_ms = int((time.monotonic() - item.enqueued_at) * 1000)
                started_at = time.monotonic()
                if item.measure is not None:
                    item.measure.queue_wait_ms += queued_ms
                    if item.measure.started_at is None:
                        item.measure.started_at = started_at
                logger.info(
                    "LLM job started: worker=%d job=%s queue_wait_ms=%d queued=%d active=%d",
                    worker_id, item.name, queued_ms, self.queued_jobs, self.active_jobs,
                )
                try:
                    value = await item.operation()
                except Exception as exc:
                    duration_ms = int((time.monotonic() - started_at) * 1000)
                    logger.exception(
                        "LLM job failed: worker=%d job=%s duration_ms=%d", worker_id, item.name, duration_ms,
                    )
                    if not item.result.done():
                        item.result.set_exception(exc)
                else:
                    duration_ms = int((time.monotonic() - started_at) * 1000)
                    logger.info(
                        "LLM job completed: worker=%d job=%s duration_ms=%d", worker_id, item.name, duration_ms,
                    )
                    if not item.result.done():
                        item.result.set_result(value)
                finally:
                    self._active -= 1
                    if item.measure is not None:
                        finished_at = time.monotonic()
                        item.measure.llm_ms += int((finished_at - started_at) * 1000)
                        item.measure.attempts += 1
                        item.measure.finished_at = finished_at
            finally:
                self._queue.task_done()