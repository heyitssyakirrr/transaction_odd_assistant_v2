"""Accounts read straight from a server folder of Parquet files, one account per file.

The folder is fixed in .env (ACCOUNT_FOLDER_PATH); staff choose only how many accounts and
which citizenship. A scan reads one column of each file (the account number) and two
columns of customer_info (account and citizenship), so listing 400 files stays fast and
the full monthly rows are read only when an account's turn comes.
"""
from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import pandas as pd
import pyarrow.parquet as pq

from app.core.reference_data import normalise_code

logger = logging.getLogger("app.account_folder")

CitizenshipChoice = Literal["malaysian", "non_malaysian", "all"]
CITIZENSHIP_CHOICES: tuple[CitizenshipChoice, ...] = ("malaysian", "non_malaysian", "all")


class AccountFolderError(RuntimeError):
    """The folder or customer_info cannot be read; the message is safe to show to staff."""


@dataclass(frozen=True)
class FolderAccount:
    path: Path
    acct_num: str
    # CITIZEN_CD of the account's latest customer_info version, e.g. "MY"; None without a profile.
    citizen_cd: str | None


@dataclass
class FolderScan:
    folder: Path
    accounts: list[FolderAccount] = field(default_factory=list)
    # (file name, reason) for files whose account number could not be read.
    unreadable: list[tuple[str, str]] = field(default_factory=list)

    def matching(self, choice: CitizenshipChoice, malaysian_code: str) -> list[FolderAccount]:
        """Accounts for a citizenship choice, in file-name order. Without a profile: only under "all"."""
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
            "total_files": len(self.accounts) + len(self.unreadable),
            "counts": {choice: len(self.matching(choice, malaysian_code)) for choice in CITIZENSHIP_CHOICES},
            "no_profile": sum(account.citizen_cd is None for account in self.accounts),
            # The codes behind the counts, so a code that is not the Malaysian one is visible on the page.
            "citizen_codes": dict(Counter(account.citizen_cd for account in self.accounts
                                          if account.citizen_cd is not None).most_common(6)),
            "unreadable": [{"file": name, "reason": reason} for name, reason in self.unreadable],
        }


def scan_folder(folder: Path, customer_info_path: Path) -> FolderScan:
    """Every *.parquet file in ``folder`` (not subfolders), sorted by name, with its account and
    citizenship. Blocking I/O: call it from a thread."""
    if not folder.is_dir():
        raise AccountFolderError(f"The account folder {folder} does not exist or is not a folder.")
    scan = FolderScan(folder=folder)
    found: list[tuple[Path, str]] = []
    for path in sorted(folder.glob("*.parquet"), key=lambda item: item.name):
        try:
            found.append((path, _account_number(path)))
        except Exception as exc:  # one bad file is listed, the rest still run
            scan.unreadable.append((path.name, str(exc)))
    citizenship = latest_citizenship(customer_info_path, {acct for _, acct in found})
    scan.accounts = [FolderAccount(path, acct, citizenship.get(acct)) for path, acct in found]
    logger.info(
        "Account folder scanned: folder=%s files=%d unreadable=%d with_profile=%d citizen_codes=%s",
        folder, len(found) + len(scan.unreadable), len(scan.unreadable), len(citizenship),
        dict(Counter(citizenship.values()).most_common(6)),
    )
    return scan


def _account_number(path: Path) -> str:
    """The one account number in the file, read from the acct_num column only."""
    names = pq.read_schema(path).names
    column = next((name for name in names if name.strip().lower() == "acct_num"), None)
    if column is None:
        raise ValueError("no acct_num column")
    values = {_identifier(value) for value in pd.read_parquet(path, columns=[column])[column].dropna()}
    if len(values) != 1:
        raise ValueError(f"expected one account number, found {len(values)}")
    return values.pop()


def latest_citizenship(customer_info_path: Path, acct_nums: set[str]) -> dict[str, str]:
    """CITIZEN_CD of each account's latest profile version (by VALID_FROM_DTTM), as plain upper-case
    text (see normalise_code); accounts with no profile or no citizenship are left out. Reads three
    columns of customer_info, once."""
    if not acct_nums:
        return {}
    try:
        table = pd.read_parquet(customer_info_path, columns=["ACCT_NUM", "CITIZEN_CD", "VALID_FROM_DTTM"])
    except FileNotFoundError as exc:
        raise AccountFolderError(f"customer_info parquet not found at {customer_info_path}") from exc
    except Exception as exc:  # pyarrow raises its own error types
        raise AccountFolderError(f"Could not read customer_info parquet: {exc}") from exc
    table = table.assign(ACCT_NUM=table["ACCT_NUM"].map(_identifier))
    table = table[table["ACCT_NUM"].isin(acct_nums)]
    table = table.assign(CITIZEN_CD=table["CITIZEN_CD"].map(normalise_code)).dropna(subset=["CITIZEN_CD"])
    latest = table.sort_values("VALID_FROM_DTTM").groupby("ACCT_NUM").tail(1)
    return dict(zip(latest["ACCT_NUM"], latest["CITIZEN_CD"]))


def _identifier(value: object) -> str:
    """An account number as text; 3062680633.0 from a float column is 3062680633."""
    text = str(value).strip()
    return text[:-2] if text.endswith(".0") and text[:-2].isdigit() else text
