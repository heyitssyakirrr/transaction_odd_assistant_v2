"""How long one LLM call took, filled in while it runs.

The work queue records when a worker took the call and how long the LLM took to answer;
the LLM client records how many tokens the answer had. A format retry is part of the same
call, so its time and tokens are added to the first attempt's.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CallMeasure:
    # Time spent in this app's queue before a worker took the call (all attempts).
    queue_wait_ms: int = 0
    # From the moment a worker sent the call to the LLM until the answer arrived (all attempts).
    # Includes any wait inside the LLM server: the app cannot see that wait separately.
    llm_ms: int = 0
    attempts: int = 0
    # Tokens the LLM wrote, from the "usage" the loader returns; None if it returned none.
    output_tokens: int | None = None
    # time.monotonic() when a worker first took the call, and when the last attempt ended.
    started_at: float | None = None
    finished_at: float | None = None
    # True when the call gave no usable answer (an HTTP error, or rejected after its retry).
    failed: bool = False

    def add_tokens(self, tokens: int | None) -> None:
        if tokens is not None:
            self.output_tokens = (self.output_tokens or 0) + tokens
