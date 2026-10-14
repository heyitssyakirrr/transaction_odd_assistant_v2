from __future__ import annotations

import csv
import io
import logging
import re
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from collections.abc import Iterable
from typing import Any

import pandas as pd

from app.core.models import CustomerProfileRecord, MonthlySummaryRow
from app.core.reference_data import normalise_code

logger = logging.getLogger("app.summary_loader")


class SummaryCsvError(ValueError):
    """Raised when the uploaded monthly-summary CSV is missing or malformed."""


class CustomerProfileError(RuntimeError):
    """Raised when the shared customer_info parquet can't be read or joined."""


_REQUIRED_COLUMNS = [
    "acct_num", "year_month", "txn_count_monthly", "pct_burst", "total_amount",
    "avg_amount", "std_amount", "max_amount", #"day_gaps", 
    "pct_trx_gap",
    "monthly_debit", "monthly_credit", "debit_count_monthly", "credit_count_monthly",
    "monthly_avg_debit", "monthly_avg_credit",
]


def parse_monthly_summary_csv(csv_content: str, expected_months: int) -> list[MonthlySummaryRow]:
    """Parse an analyst-uploaded monthly summary CSV for one account (``expected_months`` rows).

    Pure parsing: column presence/type validation and row construction only.
    No derived metrics, no scoring -- that's the LLM's job downstream.
    """
    reader = csv.DictReader(io.StringIO(csv_content))
    _check_columns(reader.fieldnames or [])
    return _monthly_rows(reader, expected_months)


def read_monthly_summary_parquet(path: Path, expected_months: int) -> list[MonthlySummaryRow]:
    """Read one account's monthly summary from a Parquet file with the same 15 columns as the CSV.
    Blocking I/O: call it from a thread."""
    try:
        table = pd.read_parquet(path)
    except Exception as exc:  # pyarrow raises its own error types
        raise SummaryCsvError(f"Could not read {path.name}: {exc}") from exc
    _check_columns(list(table.columns))
    return _monthly_rows(table.to_dict("records"), expected_months)


def _check_columns(names: list[Any]) -> None:
    present = {str(name or "").strip().lower() for name in names}
    missing = [col for col in _REQUIRED_COLUMNS if col not in present]
    if missing:
        raise SummaryCsvError(f"The file is missing required column(s): {', '.join(missing)}")


def _monthly_rows(records: Iterable[dict[Any, Any]], expected_months: int) -> list[MonthlySummaryRow]:
    """Rows for one account, checked the same way for CSV and Parquet: valid values, one account,
    exactly ``expected_months`` consecutive calendar months."""
    rows: list[MonthlySummaryRow] = []
    for row_number, raw_row in enumerate(records, start=1):
        row = {str(key or "").strip().lower(): value for key, value in raw_row.items()}
        try:
            rows.append(
                MonthlySummaryRow(
                    acct_num=_text(row["acct_num"]),
                    year_month=_text(row["year_month"]),
                    txn_count_monthly=_int(row["txn_count_monthly"]),
                    pct_burst=_float(row["pct_burst"], 4),
                    total_amount=_money(row["total_amount"]),
                    avg_amount=_money(row["avg_amount"]),
                    std_amount=_money(row["std_amount"]),
                    max_amount=_money(row["max_amount"]),
                    pct_trx_gap=_float(row["pct_trx_gap"], 2),
                    monthly_debit=_money(row["monthly_debit"]),
                    monthly_credit=_money(row["monthly_credit"]),
                    debit_count_monthly=_int(row["debit_count_monthly"]),
                    credit_count_monthly=_int(row["credit_count_monthly"]),
                    monthly_avg_debit=_money(row["monthly_avg_debit"]),
                    monthly_avg_credit=_money(row["monthly_avg_credit"]),
                )
            )
        except (KeyError, ValueError, InvalidOperation, TypeError) as exc:
            raise SummaryCsvError(f"Row {row_number}: {exc}") from exc

    if not rows:
        raise SummaryCsvError("The file contains no monthly summary rows.")

    acct_nums = {row.acct_num for row in rows}
    if len(acct_nums) != 1:
        raise SummaryCsvError(
            f"Expected all rows to belong to one account; found {len(acct_nums)}: {sorted(acct_nums)}"
        )

    rows = sorted(rows, key=lambda row: row.year_month)
    if len(rows) != expected_months:
        raise SummaryCsvError(f"Expected exactly {expected_months} monthly rows; found {len(rows)}.")
    expected = [_month_start(rows[0].year_month)]
    for _ in range(expected_months - 1):
        current = expected[-1]
        expected.append(
            current.replace(year=current.year + 1, month=1)
            if current.month == 12 else current.replace(month=current.month + 1)
        )
    if [row.year_month for row in rows] != [month.strftime("%Y%m") for month in expected]:
        raise SummaryCsvError(f"The {expected_months} monthly rows must be consecutive calendar months.")
    return rows


def review_window(rows: list[MonthlySummaryRow]) -> tuple[datetime, datetime]:
    """Derive the [start, end) review window from the supplied year_month values.

    Plumbing only: turns 'YYYY-MM' strings into UTC month boundaries so the
    customer_info overlap filter has something to compare against.
    """
    months = sorted(row.year_month for row in rows)
    return _month_start(months[0]), _month_end(months[-1])


def load_customer_profile(parquet_path: Path, acct_num: str) -> list[CustomerProfileRecord]:
    """Read every available profile version for one account.

    The Parquet filter is pushed into the reader so concurrent reviews do not
    materialise the entire customer table. The profile timeline is deliberately
    not limited to the six-month transaction window: that is the agreed review
    input contract.
    """
    try:
        table = pd.read_parquet(parquet_path, filters=[("ACCT_NUM", "==", str(acct_num))])
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
            continue
        records.append(
            CustomerProfileRecord(
                acct_num=str(row.get("ACCT_NUM")),
                customer_num=str(row.get("CUSTOMER_NUM")),
                occupation_cd=_optional_code(row.get("OCCUPATION_CD")),
                # Same cleaning as the folder filter, so b'MY' / 'MY\x00' reads as Malaysia for the AI too.
                citizen_cd=normalise_code(row.get("CITIZEN_CD")),
                indv_org_type=_optional_str(row.get("INDV_ORG_TYPE")),
                last_maint_dt=_coerce_datetime(row.get("LAST_MAINT_DT")),
                valid_from_dttm=valid_from,
                valid_to_dttm=valid_to,
            )
        )

    logger.info(
        "customer_info: acct=%s matched_full_history=%s of %s candidate rows",
        acct_num, len(records), len(account_rows),
    )
    return sorted(records, key=lambda item: item.valid_from_dttm)


def generate_case_id(acct_num: str) -> str:
    safe_acct = re.sub(r"[^a-zA-Z0-9_-]", "-", acct_num)
    return f"{safe_acct}-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"


def _decimal(value: Any) -> Decimal:
    try:
        number = Decimal(str(value).replace(",", "").strip())
    except (InvalidOperation, AttributeError) as exc:
        raise ValueError(f"invalid numeric value '{value}'") from exc
    if not number.is_finite():  # an empty Parquet cell arrives as NaN
        raise ValueError(f"invalid numeric value '{value}'")
    return number


# Values are rounded the same way whatever the source, so a CSV ("200", "2134.8333333333335") and a
# Parquet file (200.0, 2134.833333333333) give the AI exactly the same figures.
def _money(value: Any) -> Decimal:
    """An amount in RM, to the cent."""
    return _decimal(value).quantize(Decimal("0.01"))


def _float(value: Any, places: int) -> float:
    """A rate: pct_burst to 4 places (0.1% steps when shown as a percentage), gaps in days to 2."""
    return float(_decimal(value).quantize(Decimal(1).scaleb(-places)))


def _int(value: Any) -> int:
    """A count; 12.0 from a Parquet float column is 12, 12.5 is an error."""
    number = _decimal(value)
    if number != number.to_integral_value():
        raise ValueError(f"invalid count '{value}'")
    return int(number)


def _text(value: Any) -> str:
    """An identifier such as acct_num or year_month; 202604.0 from a Parquet float column is 202604."""
    text = str(value).strip()
    return text[:-2] if re.fullmatch(r"\d+\.0", text) else text


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


def _optional_code(value: Any) -> str | None:
    """Preserve reference-code semantics when Parquet exposes nullable numerics.

    Pandas may represent a code such as 12 as 12.0 when the column contains
    nulls. The occupation reference CSV stores `12`, so normalise only this
    numeric artefact while leaving genuine string codes (including leading
    zeroes) unchanged.
    """
    text = _optional_str(value)
    if text is None:
        return None
    if re.fullmatch(r"[+-]?\d+\.0+", text):
        return text.split(".", 1)[0]
    return text
