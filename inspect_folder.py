"""Which accounts in the account folder can be found in customer_info, and with which citizenship?

Usage (from the project root, with your .env in place):
    python test/inspect_folder.py
    python test/inspect_folder.py --folder /pythonITD/dwh_312/for_you/high_risk --customer-info /path/customer_info.parquet

It prints a short summary (no full account numbers) and writes a CSV with one row per
folder file, so you can filter it yourself:
    data/folder_check.csv: file, acct_num, rows, months, in_customer_info,
                           in_customer_info_ignoring_leading_zeros, profile_versions,
                           latest_citizen_cd, citizen_cd_as_stored, what_the_app_counts_it_as
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import Settings  # noqa: E402
from app.core.account_folder import scan_folder  # noqa: E402


def masked(value: object) -> str:
    """The type and shape of a value without showing it in full: str '400...22' (10 chars)."""
    text = repr(value) if isinstance(value, bytes) else str(value)
    shown = text if len(text) <= 6 else f"{text[:3]}...{text[-2:]}"
    return f"{type(value).__name__} {shown!r} ({len(text)} chars)"


def main() -> None:
    settings = Settings()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--folder", default=str(settings.account_folder or ""), help="default: ACCOUNT_FOLDER_PATH")
    parser.add_argument("--customer-info", default=str(settings.customer_info_path),
                        help="default: CUSTOMER_INFO_PARQUET_PATH")
    parser.add_argument("--out", default="data/folder_check.csv")
    args = parser.parse_args()
    folder, info_path = Path(args.folder), Path(args.customer_info)
    print(f"Folder:        {folder}")
    print(f"customer_info: {info_path}")
    print(f"Malaysian code (MALAYSIAN_CITIZEN_CD): {settings.malaysian_citizen_cd!r}\n")

    # 1. The folder files, read the way the analysis reads them.
    files = sorted(folder.glob("*.parquet"), key=lambda path: path.name)
    others = [path.name for path in folder.iterdir() if path.is_file() and path.suffix != ".parquet"]
    print(f"Folder: {len(files)} .parquet files" + (f", {len(others)} other files (ignored)" if others else ""))
    rows: list[dict[str, object]] = []
    folder_raw_types: Counter[str] = Counter()
    for path in files:
        row: dict[str, object] = {"file": path.name}
        try:
            names = pq.read_schema(path).names
            column = next((name for name in names if name.strip().lower() == "acct_num"), None)
            if column is None:
                raise ValueError(f"no acct_num column (columns: {', '.join(names)})")
            frame = pd.read_parquet(path)
            values = frame[column].dropna().unique()
            folder_raw_types[type(values[0]).__name__ if len(values) else "none"] += 1
            month_col = next((name for name in names if name.strip().lower() == "year_month"), None)
            row.update(acct_num=str(values[0]).strip() if len(values) == 1 else f"{len(values)} different values",
                       raw=values[0] if len(values) else None, rows=len(frame),
                       months=frame[month_col].nunique() if month_col else "no year_month")
        except Exception as exc:  # report and carry on
            row.update(acct_num=f"UNREADABLE: {exc}", raw=None, rows="", months="")
        rows.append(row)
    if rows and rows[0]["raw"] is not None:
        print(f"  first file's acct_num as stored: {masked(rows[0]['raw'])}")
    print(f"  acct_num stored as: {dict(folder_raw_types)}")
    print(f"  rows per file: {dict(Counter(row['rows'] for row in rows).most_common(4))}")
    print(f"  months per file: {dict(Counter(row['months'] for row in rows).most_common(4))}\n")

    # 2. customer_info, three columns only.
    info = pd.read_parquet(info_path, columns=["ACCT_NUM", "CITIZEN_CD", "VALID_FROM_DTTM"])
    print(f"customer_info: {len(info):,} rows, {info['ACCT_NUM'].nunique():,} accounts")
    print(f"  ACCT_NUM dtype {info['ACCT_NUM'].dtype}; first value as stored: {masked(info['ACCT_NUM'].iloc[0])}")
    print(f"  CITIZEN_CD dtype {info['CITIZEN_CD'].dtype}; first value as stored: {info['CITIZEN_CD'].dropna().iloc[0]!r}")
    print("  CITIZEN_CD most common values, exactly as stored (repr shows spaces, bytes, case):")
    for value, count in info["CITIZEN_CD"].map(repr).value_counts().head(8).items():
        print(f"    {value:<16} {count:,}")
    print()

    # 3. Match each folder account, exactly and ignoring leading zeros; latest profile version's citizenship.
    info = info.assign(_key=info["ACCT_NUM"].map(lambda value: str(value).strip()))
    info = info.assign(_key_nozero=info["_key"].str.lstrip("0"))
    by_key = info.groupby("_key").size()
    nozero_keys = set(info["_key_nozero"])
    latest = info.sort_values("VALID_FROM_DTTM").groupby("_key").tail(1).set_index("_key")["CITIZEN_CD"]

    # 4. What the app itself counts each account as (the same code the upload page uses).
    app_view: dict[str, str] = {}
    try:
        scan = scan_folder(folder, info_path)
        for account in scan.accounts:
            app_view[account.path.name] = (
                "no profile" if account.citizen_cd is None
                else "Malaysian" if account.citizen_cd == settings.malaysian_citizen_cd else "Non-Malaysian")
        for name, reason in scan.unreadable:
            app_view[name] = f"unreadable: {reason}"
    except Exception as exc:
        print(f"The app's own scan failed: {exc}\n")

    for row in rows:
        key = str(row["acct_num"])
        stored = latest.get(key)
        row.update(
            in_customer_info="yes" if key in by_key.index else "no",
            in_customer_info_ignoring_leading_zeros="yes" if key.lstrip("0") in nozero_keys else "no",
            profile_versions=int(by_key.get(key, 0)),
            latest_citizen_cd="" if stored is None or pd.isna(stored) else str(stored).strip(),
            citizen_cd_as_stored="" if stored is None else repr(stored),
            what_the_app_counts_it_as=app_view.get(str(row["file"]), ""),
        )
        row.pop("raw")

    found = sum(row["in_customer_info"] == "yes" for row in rows)
    found_nozero = sum(row["in_customer_info_ignoring_leading_zeros"] == "yes" for row in rows)
    print(f"Folder accounts found in customer_info: {found} of {len(rows)} exactly; "
          f"{found_nozero} ignoring leading zeros")
    print(f"Latest citizenship of the found accounts: "
          f"{dict(Counter(row['latest_citizen_cd'] or '(empty)' for row in rows if row['in_customer_info'] == 'yes'))}")
    print(f"What the app counts them as: {dict(Counter(row['what_the_app_counts_it_as'] for row in rows))}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["file"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nOne row per file: {out.resolve()}")
    print("Filter in_customer_info = yes (and latest_citizen_cd = MY) to find the accounts you can test.")


if __name__ == "__main__":
    main()