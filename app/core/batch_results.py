"""One CSV per batch with a row per finished account, written as each account finishes.

It answers the folder test directly: the rule-based program called every account high risk;
what did this assessment find? Rows are appended as accounts finish (several accounts run at
once, so rows are in finishing order; position and list_position give the list order), so the
results of a long run survive an application restart. The expected risk is written here only; it is
never part of what the LLM sees.
"""
from __future__ import annotations

import asyncio
import csv
from pathlib import Path
from typing import Any

from app.core.batch_runner import Batch, BatchFile

COLUMNS = (
    "position", "list_position", "file", "acct_num", "citizen_cd", "expected_risk", "risk_level", "matches_expected",
    "decision", "status", "error", "calls_1_to_3_seconds", "total_seconds", "report_html",
)


class BatchResultsLog:
    def __init__(self, directory: Path) -> None:
        self._directory = directory
        # Several accounts can finish at the same moment; one row is written at a time.
        self._write_lock = asyncio.Lock()

    def path_for(self, batch: Batch) -> Path:
        return self._directory / f"{batch.created_at.strftime('%Y%m%d-%H%M%S')}-{batch.batch_id[:12]}.csv"

    def start(self, batch: Batch) -> None:
        """Create the batch's CSV with its header and remember its name on the batch."""
        self._directory.mkdir(parents=True, exist_ok=True)
        path = self.path_for(batch)
        with path.open("w", newline="", encoding="utf-8") as handle:
            csv.writer(handle).writerow(COLUMNS)
        batch.results_csv = path.name

    async def record(self, batch: Batch, position: int, item: BatchFile) -> None:
        async with self._write_lock:
            await asyncio.to_thread(self._append, batch, position, item)

    def _append(self, batch: Batch, position: int, item: BatchFile) -> None:
        with self.path_for(batch).open("a", newline="", encoding="utf-8") as handle:
            csv.writer(handle).writerow(row(batch, position, item))

    def resolve(self, name: str) -> Path | None:
        candidate = (self._directory / Path(name).name).resolve()
        if candidate.parent != self._directory.resolve() or candidate.suffix != ".csv":
            return None
        return candidate if candidate.exists() else None


def row(batch: Batch, position: int, item: BatchFile) -> list[Any]:
    result = item.result or {}
    timing = result.get("timing") or {}
    risk = result.get("risk_level", "")
    expected = batch.expected_risk or ""
    return [
        position, item.list_position or "", item.filename, item.acct_num or result.get("acct_num", ""), item.citizen_cd or "", expected,
        risk, ("yes" if risk == expected else "no") if expected and risk else "", result.get("decision", ""),
        item.status, item.error or "", timing.get("calls_1_to_3_seconds", ""), timing.get("total_seconds", ""),
        result.get("report_html", ""),
    ]
