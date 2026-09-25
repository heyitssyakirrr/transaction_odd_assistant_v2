from __future__ import annotations

"""Reference-data lookups for translating coded customer_info fields into
labels the LLM can reason about.

Kept separate from prompts.py and analysis_service.py on purpose: this data
changes on compliance's schedule, not the application's, and should never
require touching prompt text or orchestration logic. The LLM is never shown
these tables -- only the resolved label for the one row it's looking at, so
the payload stays small regardless of how large the CSV grows.
"""

import csv
import logging
from functools import lru_cache
from pathlib import Path

logger = logging.getLogger("app.reference_data")

_OCCUPATION_CODE_COLUMN = "CODE"
_OCCUPATION_LABEL_COLUMN = "D_CUST_OCCUPAT"

# ISO-3166-alpha-2 codes are already fairly self-explanatory, but spelling
# them out removes any ambiguity for the model and costs very little. Small
# and static enough to keep inline rather than as a CSV.
CITIZEN_CODE_MAP: dict[str, str] = {
    "MY": "Malaysia",
    "SG": "Singapore",
    "ID": "Indonesia",
}


@lru_cache(maxsize=8)
def _load_occupation_map(csv_path: str) -> dict[str, str]:
    """Load CODE -> D_CUST_OCCUPAT from the occupation reference CSV.

    Cached per path so the file is read once per process (the file is
    small and static -- restart the process after updating it). Missing
    file or missing columns fails SOFT: log loudly and return an empty
    map, so a reference-data gap degrades individual occupation labels to
    "unmapped" rather than taking analysis down entirely.
    """
    path = Path(csv_path)
    if not path.exists():
        logger.error(
            "Occupation code CSV not found at %s -- every occupation will show as unmapped "
            "until OCCUPATION_CODE_CSV_PATH points at a real file.",
            path,
        )
        return {}
    try:
        with path.open(encoding="utf-8-sig", newline="") as fh:
            reader = csv.DictReader(fh)
            fieldnames = {(name or "").strip() for name in (reader.fieldnames or [])}
            missing = {_OCCUPATION_CODE_COLUMN, _OCCUPATION_LABEL_COLUMN} - fieldnames
            if missing:
                logger.error(
                    "Occupation code CSV at %s is missing column(s): %s -- every occupation will "
                    "show as unmapped.",
                    path, ", ".join(sorted(missing)),
                )
                return {}
            mapping = {
                row[_OCCUPATION_CODE_COLUMN].strip(): row[_OCCUPATION_LABEL_COLUMN].strip()
                for row in reader
                if (row.get(_OCCUPATION_CODE_COLUMN) or "").strip()
            }
            logger.info("Loaded %s occupation code(s) from %s", len(mapping), path)
            return mapping
    except OSError as exc:
        logger.error("Failed to read occupation code CSV at %s: %s", path, exc)
        return {}


def resolve_occupation(code: str | None, csv_path: Path | str) -> str | None:
    if code is None:
        return None
    return _load_occupation_map(str(csv_path)).get(code, f"Unmapped occupation code ({code})")


def resolve_citizenship(code: str | None) -> str | None:
    if code is None:
        return None
    # Fall back to the raw code -- still interpretable on its own (MY, SG...)
    # even for a value this table doesn't cover yet.
    return CITIZEN_CODE_MAP.get(code, code)
