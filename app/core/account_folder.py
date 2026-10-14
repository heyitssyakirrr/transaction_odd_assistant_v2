"""Accounts read straight from a server folder of Parquet files, one account per file.

The folder is fixed in .env (ACCOUNT_FOLDER_PATH); staff choose only a range of accounts and
which citizenship. A scan reads each file's account number from its Parquet footer (no data),
several files at a time, and four columns of customer_info. Accounts without a usable
occupation (no profile, no code, or a code not in the occupation list) are left out: the
summary's occupation judgement needs one. The result is kept until a file in the folder,
customer_info or the occupation list changes, so opening the folder tab again, or starting a
run, does not scan again. The full monthly rows are read only when an account's turn comes.
"""
from __future__ import annotations

import logging
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import pandas as pd
import pyarrow.parquet as pq

from app.core.reference_data import has_occupation_label, normalise_code, occupation_code, occupation_list_size

logger = logging.getLogger("app.account_folder")

CitizenshipChoice = Literal["malaysian", "non_malaysian", "all"]
CITIZENSHIP_CHOICES: tuple[CitizenshipChoice, ...] = ("malaysian", "non_malaysian", "all")


class AccountFolderError(RuntimeError):
    """The folder or customer_info cannot be read; the message is safe to show to staff."""


@dataclass(frozen=True)
class FolderAccount:
    path: Path
    acct_num: str
    # From the account's latest customer_info version: CITIZEN_CD (e.g. "MY") and OCCUPATION_CD.
    citizen_cd: str | None
    occupation_cd: str | None


# Why an account is left out of every choice, as shown to staff.
NO_PROFILE = "no customer profile"
NO_OCCUPATION = "no occupation code"
UNKNOWN_OCCUPATION = "occupation code not in the occupation list"


@dataclass
class FolderScan:
    folder: Path
    # Accounts that can be analysed (they have a usable occupation), in file-name order.
    accounts: list[FolderAccount] = field(default_factory=list)
    # Accounts left out, with the reason (see NO_PROFILE and the others).
    left_out: list[tuple[FolderAccount, str]] = field(default_factory=list)
    # (file name, reason) for files whose account number could not be read.
    unreadable: list[tuple[str, str]] = field(default_factory=list)

    def matching(self, choice: CitizenshipChoice, malaysian_code: str) -> list[FolderAccount]:
        """Accounts for a citizenship choice, in file-name order; an account with no citizenship
        code is only under "all"."""
        if choice == "all":
            return list(self.accounts)
        malaysian = choice == "malaysian"
        malaysian_code = normalise_code(malaysian_code) or ""
        return [
            account for account in self.accounts
            if account.citizen_cd is not None and (account.citizen_cd == malaysian_code) == malaysian
        ]

    def summary(self, malaysian_code: str) -> dict[str, object]:
        return {
            "folder": str(self.folder),
            "total_files": len(self.accounts) + len(self.left_out) + len(self.unreadable),
            "counts": {choice: len(self.matching(choice, malaysian_code)) for choice in CITIZENSHIP_CHOICES},
            "left_out": dict(Counter(reason for _, reason in self.left_out)),
            # The codes behind the counts, so a code that is not the Malaysian one is visible on the page.
            "citizen_codes": dict(Counter(account.citizen_cd for account in self.accounts
                                          if account.citizen_cd is not None).most_common(6)),
            "unreadable": [{"file": name, "reason": reason} for name, reason in self.unreadable],
        }


# Files read at the same time during a scan: the work is waiting on storage, not the CPU.
_SCAN_THREADS = 16
# The last scan, with what it was made from; reused while nothing it read has changed.
_cache_lock = threading.Lock()
_cache: tuple[tuple, FolderScan] | None = None


def scan_folder(folder: Path, customer_info_path: Path, occupation_list_path: Path) -> FolderScan:
    """Every *.parquet file in ``folder`` (not subfolders), sorted by name, with its account,
    citizenship and occupation. Reuses the previous scan when no file in the folder, customer_info
    or the occupation list has changed (same names, sizes and modification times). Blocking I/O:
    call it from a thread."""
    global _cache
    if not folder.is_dir():
        raise AccountFolderError(f"The account folder {folder} does not exist or is not a folder.")
    if not occupation_list_size(occupation_list_path):
        # Without the list every occupation would look unknown and every account would be left out.
        raise AccountFolderError(
            f"The occupation list was not found or is empty at {occupation_list_path} (OCCUPATION_CODE_CSV_PATH).")
    paths = sorted(folder.glob("*.parquet"), key=lambda item: item.name)
    key = (str(folder), _fingerprint(paths), _fingerprint([customer_info_path, occupation_list_path]))
    with _cache_lock:  # one scan at a time; a second request waits and reuses it
        if _cache is not None and _cache[0] == key:
            return _cache[1]
        scan = _scan(folder, customer_info_path, occupation_list_path, paths)
        _cache = (key, scan)
        return scan


def _fingerprint(paths: list[Path]) -> tuple:
    """Names, sizes and modification times: one directory listing, no file contents."""
    stats = []
    for path in paths:
        try:
            stat = path.stat()
            stats.append((path.name, stat.st_size, stat.st_mtime_ns))
        except OSError:
            stats.append((path.name, None, None))
    return tuple(stats)


def _scan(folder: Path, customer_info_path: Path, occupation_list_path: Path, paths: list[Path]) -> FolderScan:
    scan = FolderScan(folder=folder)
    found: list[tuple[Path, str]] = []

    def read(path: Path) -> tuple[Path, str | None, str | None]:
        try:
            return path, _account_number(path), None
        except Exception as exc:  # one bad file is listed, the rest still run
            return path, None, str(exc)

    with ThreadPoolExecutor(max_workers=_SCAN_THREADS) as pool:
        for path, acct, problem in pool.map(read, paths):  # map keeps file-name order
            if acct is None:
                scan.unreadable.append((path.name, problem or "unreadable"))
            else:
                found.append((path, acct))
    profiles = latest_profiles(customer_info_path, {acct for _, acct in found})
    for path, acct in found:
        citizen_cd, occupation_cd = profiles.get(acct, (None, None))
        account = FolderAccount(path, acct, citizen_cd, occupation_cd)
        reason = (NO_PROFILE if acct not in profiles else NO_OCCUPATION if occupation_cd is None
                  else None if has_occupation_label(occupation_cd, occupation_list_path) else UNKNOWN_OCCUPATION)
        if reason is None:
            scan.accounts.append(account)
        else:
            scan.left_out.append((account, reason))
    logger.info(
        "Account folder scanned: folder=%s files=%d usable=%d left_out=%s unreadable=%d citizen_codes=%s",
        folder, len(paths), len(scan.accounts), dict(Counter(reason for _, reason in scan.left_out)),
        len(scan.unreadable), dict(Counter(account.citizen_cd for account in scan.accounts).most_common(6)),
    )
    return scan


def _account_number(path: Path) -> str:
    """The one account number in the file. Parquet keeps each column's smallest and largest value in
    the file footer; when they are equal in every row group, that is the account, with no data read.
    Otherwise (no statistics, or two accounts) the acct_num column itself is read."""
    parquet = pq.ParquetFile(path)
    names = parquet.schema_arrow.names
    column = next((name for name in names if name.strip().lower() == "acct_num"), None)
    if column is None:
        raise ValueError("no acct_num column")
    if (from_footer := _single_value_in_footer(parquet, names.index(column))) is not None:
        return from_footer
    values = {_identifier(value) for value in parquet.read(columns=[column]).column(0).to_pylist() if value is not None}
    if len(values) != 1:
        raise ValueError(f"expected one account number, found {len(values)}")
    return values.pop()


def _single_value_in_footer(parquet: pq.ParquetFile, column_index: int) -> str | None:
    """The column's only value when the footer statistics show one (min == max in every row group)."""
    values = set()
    metadata = parquet.metadata
    for group in range(metadata.num_row_groups):
        stats = metadata.row_group(group).column(column_index).statistics
        if stats is None or not stats.has_min_max or stats.null_count:
            return None
        values.update((_identifier(stats.min), _identifier(stats.max)))
    return values.pop() if len(values) == 1 else None


def latest_profiles(customer_info_path: Path, acct_nums: set[str]) -> dict[str, tuple[str | None, str | None]]:
    """(CITIZEN_CD, OCCUPATION_CD) of each account's latest profile version by VALID_FROM_DTTM, the
    version the analysis reads too, cleaned like the analysis cleans them. Accounts not in
    customer_info are left out. Reads four columns of customer_info, once."""
    if not acct_nums:
        return {}
    try:
        table = pd.read_parquet(customer_info_path,
                                columns=["ACCT_NUM", "CITIZEN_CD", "OCCUPATION_CD", "VALID_FROM_DTTM"])
    except FileNotFoundError as exc:
        raise AccountFolderError(f"customer_info parquet not found at {customer_info_path}") from exc
    except Exception as exc:  # pyarrow raises its own error types
        raise AccountFolderError(f"Could not read customer_info parquet: {exc}") from exc
    table = table.assign(ACCT_NUM=table["ACCT_NUM"].map(_identifier))
    table = table[table["ACCT_NUM"].isin(acct_nums)]
    # The analysis skips versions without a start date (load_customer_profile), so this does too.
    latest = table.dropna(subset=["VALID_FROM_DTTM"]).sort_values("VALID_FROM_DTTM").groupby("ACCT_NUM").tail(1)
    return {
        acct: (normalise_code(citizen), occupation_code(occupation))
        for acct, citizen, occupation in zip(latest["ACCT_NUM"], latest["CITIZEN_CD"], latest["OCCUPATION_CD"])
    }


def _identifier(value: object) -> str:
    """An account number as text; 3062680633.0 from a float column is 3062680633, b'306...' is 306..."""
    if isinstance(value, (bytes, bytearray)):
        value = bytes(value).decode("utf-8", errors="replace")
    text = str(value).replace("\x00", "").strip()
    return text[:-2] if text.endswith(".0") and text[:-2].isdigit() else text
