"""Persistent, bounded application logging for review diagnostics."""

from __future__ import annotations

import logging
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path


LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"
LOG_FILENAME = "transaction_summary.log"


def configure_persistent_logging(log_directory: Path, retention_days: int) -> Path | None:
    """Add one UTC-daily rotating UTF-8 file handler to the root logger.

    The function is idempotent because Uvicorn can import the application more
    than once in some deployment modes. If the mounted directory is not
    writable, console logging remains available and application startup does
    not fail.
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    formatter = logging.Formatter(LOG_FORMAT)
    if not root_logger.handlers:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)

    log_path = log_directory / LOG_FILENAME
    try:
        log_directory.mkdir(parents=True, exist_ok=True)
        resolved_path = log_path.resolve()
        for handler in root_logger.handlers:
            if isinstance(handler, TimedRotatingFileHandler) and Path(handler.baseFilename).resolve() == resolved_path:
                return log_path

        file_handler = TimedRotatingFileHandler(
            log_path,
            when="midnight",
            interval=1,
            backupCount=retention_days,
            encoding="utf-8",
            utc=True,
        )
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
        return log_path
    except OSError as exc:
        root_logger.warning("Persistent log file is unavailable at %s: %s", log_path, exc)
        return None
