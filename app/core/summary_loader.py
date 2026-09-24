from __future__ import annotations

import csv
import io
import logging
import re
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

import pandas as pd

from app.core.models import CustomerProfileRecord, MonthlySummaryRow

logger = logging.getLogger("app.summary_loader")


class SummaryCsvError(ValueError):
    """Raised when the uploaded monthly-summary CSV is missing or malformed."""


class CustomerProfileError(RuntimeError):
    """Raised when the shared customer_info parquet can't be read or joined."""


_REQUIRED_COLUMNS = [
    "acct_num", "year_month", "txn_count_monthly", "pct_burst", "total_amount",
    "avg_amount", "std_amount", "max_amount", "day_gaps", "pct_trx_gap",
    "monthly_debit", "monthly_credit", "debit_count_monthly", "credit_count_monthly",
    "monthly_avg_debit", "monthly_avg_credit",
]


def parse_monthly_summary_csv(csv_content: str) -> list[MonthlySummaryRow]:
    """Parse the analyst-uploaded 6-row monthly summary CSV for one account.

    Pure parsing: column presence/type validation and row construction only.
    No derived metrics, no scoring -- that's the LLM's job downstream.
    """
    reader = csv.DictReader(io.StringIO(csv_content))
    fieldnames = reader.fieldnames or []
    missing = [col for col in _REQUIRED_COLUMNS if col not in fieldnames]
    if missing:
        raise SummaryCsvError(f"CSV is missing required column(s): {', '.join(missing)}")

    rows: list[MonthlySummaryRow] = []
    for row_number, raw_row in enumerate(reader, start=1):
        try:
            rows.append(
                MonthlySummaryRow(
                    acct_num=str(raw_row["acct_num"]).strip(),
                    year_month=str(raw_row["year_month"]).strip(),
                    txn_count_monthly=int(raw_row["txn_count_monthly"]),
                    pct_burst=float(raw_row["pct_burst"]),
                    total_amount=_decimal(raw_row["total_amount"]),
                    avg_amount=_decimal(raw_row["avg_amount"]),
                    std_amount=_decimal(raw_row["std_amount"]),
                    max_amount=_decimal(raw_row["max_amount"]),
                    day_gaps=float(raw_row["day_gaps"]),
                    pct_trx_gap=float(raw_row["pct_trx_gap"]),
                    monthly_debit=_decimal(raw_row["monthly_debit"]),
                    monthly_credit=_decimal(raw_row["monthly_credit"]),
                    debit_count_monthly=int(raw_row["debit_count_monthly"]),
                    credit_count_monthly=int(raw_row["credit_count_monthly"]),
                    monthly_avg_debit=_decimal(raw_row["monthly_avg_debit"]),
                    monthly_avg_credit=_decimal(raw_row["monthly_avg_credit"]),
                )
            )
        except (KeyError, ValueError, InvalidOperation, TypeError) as exc:
            raise SummaryCsvError(f"Row {row_number}: {exc}") from exc

    if not rows:
        raise SummaryCsvError("The CSV contains no monthly summary rows.")

    acct_nums = {row.acct_num for row in rows}
    if len(acct_nums) != 1:
        raise SummaryCsvError(
            f"Expected all rows to belong to one account; found {len(acct_nums)}: {sorted(acct_nums)}"
        )

    return sorted(rows, key=lambda row: row.year_month)


def review_window(rows: list[MonthlySummaryRow]) -> tuple[datetime, datetime]:
    """Derive the [start, end) review window from the supplied year_month values.

    Plumbing only: turns 'YYYY-MM' strings into UTC month boundaries so the
    customer_info overlap filter has something to compare against.
    """
    months = sorted(row.year_month for row in rows)
    return _month_start(months[0]), _month_end(months[-1])


def load_customer_profile(
    parquet_path: Path, acct_num: str, window_start: datetime, window_end: datetime
) -> list[CustomerProfileRecord]:
    """Read the shared customer_info SCD2 table and keep every row for this
    account whose validity period overlaps [window_start, window_end).

    Overlap filter (membership check, not math):
        VALID_FROM_DTTM <= window_end AND
        (VALID_TO_DTTM IS NULL OR VALID_TO_DTTM >= window_start)

    A profile change coinciding with a transaction-pattern change in the same
    window is a signal the LLM should be able to see -- so every overlapping
    version is kept, not just the current one.
    """
    try:
        table = pd.read_parquet(parquet_path)
    except FileNotFoundError as exc:
        raise CustomerProfileError(f"customer_info parquet not found at {parquet_path}") from exc
    except Exception as exc:  # pyarrow/fastparquet raise their own error types
        raise CustomerProfileError(f"Failed to read customer_info parquet: {exc}") from exc

    if "ACCT_NUM" not in table.columns:
        raise CustomerProfileError("customer_info parquet is missing the ACCT_NUM column.")

    account_rows = table[table["ACCT_NUM"].astype(str) == str(acct_num)]

    records: list[CustomerProfileRecord] = []
    for _, row in account_rows.iterrows():
        valid_from = _coerce_datetime(row.get("VALID_FROM_DTTM"))
        valid_to = _coerce_datetime(row.get("VALID_TO_DTTM"))
        if valid_from is None:
            continue  # unusable row: no start of validity to anchor the overlap check
        if not (valid_from <= window_end and (valid_to is None or valid_to >= window_start)):
            continue
        records.append(
            CustomerProfileRecord(
                acct_num=str(row.get("ACCT_NUM")),
                customer_num=str(row.get("CUSTOMER_NUM")),
                occupation_cd=_optional_str(row.get("OCCUPATION_CD")),
                citizen_cd=_optional_str(row.get("CITIZEN_CD")),
                indv_org_type=_optional_str(row.get("INDV_ORG_TYPE")),
                last_maint_dt=_coerce_datetime(row.get("LAST_MAINT_DT")),
                valid_from_dttm=valid_from,
                valid_to_dttm=valid_to,
            )
        )

    logger.info(
        "customer_info: acct=%s window=%s..%s matched=%s of %s candidate rows",
        acct_num, window_start.date(), window_end.date(), len(records), len(account_rows),
    )
    return sorted(records, key=lambda item: item.valid_from_dttm)


def generate_case_id(acct_num: str) -> str:
    safe_acct = re.sub(r"[^a-zA-Z0-9_-]", "-", acct_num)
    return f"{safe_acct}-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"


def _decimal(value: Any) -> Decimal:
    try:
        return Decimal(str(value).replace(",", "").strip())
    except (InvalidOperation, AttributeError) as exc:
        raise ValueError(f"invalid numeric value '{value}'") from exc


def _month_start(year_month: str) -> datetime:
    try:
        return datetime.strptime(year_month, "%Y%m").replace(tzinfo=timezone.utc)
    except ValueError as exc:
        raise SummaryCsvError(f"invalid year_month '{year_month}'; expected YYYYMM, e.g. 202604") from exc


def _month_end(year_month: str) -> datetime:
    start = _month_start(year_month)
    return start.replace(year=start.year + 1, month=1) if start.month == 12 else start.replace(month=start.month + 1)


def _coerce_datetime(value: Any) -> datetime | None:
    if value is None or pd.isna(value):
        return None
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return ts.to_pydatetime()


def _optional_str(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    return text or None