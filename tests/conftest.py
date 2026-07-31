"""Shared pytest fixtures for the SEC Value Screen test suite.

Provides helpers to construct synthetic SEC DERA-format data (sub.txt /
num.txt / tag.txt) in temporary directories, plus common fixtures for
TTM scenarios and edge cases.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Union

import duckdb
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Low-level helpers: write synthetic SEC text files
# ---------------------------------------------------------------------------

def _tsv_line(fields: List[str]) -> str:
    return "\t".join(fields) + "\n"


def write_sub_txt(path: Path, rows: List[Dict[str, str]]) -> None:
    """Write a sub.txt file from a list of row dicts. Columns not provided
    default to empty strings."""
    columns = [
        "adsh", "cik", "name", "sic", "countryba", "stprba", "cityba",
        "zipba", "bas1", "bas2", "baph", "countryma", "stprma", "cityma",
        "zipma", "mas1", "mas2", "countryinc", "stprinc", "ein",
        "former", "changed", "afs", "wksi", "fye", "form", "period",
        "fy", "fp", "filed", "accepted", "prevrpt", "detail", "instance",
        "nciks", "aciks",
    ]
    with open(path, "w") as f:
        f.write(_tsv_line(columns))
        for row in rows:
            f.write(_tsv_line([row.get(c, "") for c in columns]))


def write_num_txt(path: Path, rows: List[Dict[str, str]]) -> None:
    """Write a num.txt file from a list of row dicts."""
    columns = [
        "adsh", "tag", "version", "coreg", "ddate", "qtrs", "uom",
        "value", "footnote",
    ]
    with open(path, "w") as f:
        f.write(_tsv_line(columns))
        for row in rows:
            f.write(_tsv_line([row.get(c, "") for c in columns]))


def write_tag_txt(path: Path, rows: List[Dict[str, str]]) -> None:
    """Write a tag.txt file from a list of row dicts."""
    columns = [
        "tag", "version", "custom", "abstract", "datatype",
        "iord", "crdr", "tlabel", "doc",
    ]
    with open(path, "w") as f:
        f.write(_tsv_line(columns))
        for row in rows:
            f.write(_tsv_line([row.get(c, "") for c in columns]))


def write_pre_txt(path: Path, rows: List[Dict[str, str]]) -> None:
    """Write a pre.txt file (SEC presentation linkbase) from a list of row
    dicts.  Columns not provided default to empty strings."""
    columns = [
        "adsh", "report", "line", "stmt", "inpth",
        "tag", "version", "plabel", "negating",
    ]
    with open(path, "w") as f:
        f.write(_tsv_line(columns))
        for row in rows:
            f.write(_tsv_line([row.get(c, "") for c in columns]))


# ---------------------------------------------------------------------------
# Fixture: scratch DuckDB connection (in-memory, survives across test cases)
# ---------------------------------------------------------------------------

@pytest.fixture
def db_con():
    """Return a fresh in-memory DuckDB connection."""
    return duckdb.connect()


# ---------------------------------------------------------------------------
# Fixtures for common TTM/ratio scenarios
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_data_dir(tmp_path: Path) -> Path:
    """Empty temp directory for synthetic SEC data."""
    return tmp_path


def make_ttm_company_data(base_dir: Path) -> str:
    """Build a synthetic dataset for TTM computation testing.

    One company (CIK 12345) with:
    - FY2023 10-K (period 20231231, filed 20240215, Revenues=1000, OpInc=150, NI=100)
    - Q1 2023 10-Q (period 20230331, fp=Q1, qtrs=1, Revenues=240, OpInc=35, NI=25)
    - Q1 2024 10-Q (period 20240331, fp=Q1, qtrs=1, Revenues=260, OpInc=40, NI=30)
    plus balance-sheet fields.

    TTM Revenues = 1000 + 260 - 240 = 1020
    TTM OperatingIncomeLoss = 150 + 40 - 35 = 155
    """
    adsh_10k = "0000012345-24-000001"
    adsh_q1_23 = "0000012345-23-000002"
    adsh_q1_24 = "0000012345-24-000003"

    write_sub_txt(base_dir / "sub.txt", [
        {"adsh": adsh_10k, "cik": "12345", "name": "TEST CORP", "sic": "7370",
         "form": "10-K", "period": "20231231", "fy": "2023", "fp": "FY",
         "filed": "20240215"},
        {"adsh": adsh_q1_23, "cik": "12345", "name": "TEST CORP", "sic": "7370",
         "form": "10-Q", "period": "20230331", "fy": "2023", "fp": "Q1",
         "filed": "20230501"},
        {"adsh": adsh_q1_24, "cik": "12345", "name": "TEST CORP", "sic": "7370",
         "form": "10-Q", "period": "20240331", "fy": "2024", "fp": "Q1",
         "filed": "20240501"},
    ])

    # Build num rows for each filing
    num_rows = []

    def add_fact(adsh: str, tag: str, qtrs: str, ddate: str, value: str,
                 coreg: str = "", uom: str = "USD"):
        num_rows.append({
            "adsh": adsh, "tag": tag, "version": "us-gaap/2023",
            "coreg": coreg, "ddate": ddate, "qtrs": qtrs, "uom": uom,
            "value": value, "footnote": "",
        })

    # --- 10-K FY2023 facts (ddate = 20231231) ---
    # Flow fields: qtrs=4
    for tag, val in [("Revenues", "1000000000"), ("OperatingIncomeLoss", "150000000"),
                     ("NetIncomeLoss", "100000000"), ("InterestExpense", "5000000"),
                     ("IncomeTaxExpenseBenefit", "20000000"),
                     ("NetCashProvidedByUsedInOperatingActivities", "120000000"),
                     ("PaymentsToAcquirePropertyPlantAndEquipment", "30000000")]:
        add_fact(adsh_10k, tag, "4", "20231231", val)
    # Balance sheet: qtrs=0
    for tag, val in [("Assets", "2000000000"), ("AssetsCurrent", "500000000"),
                     ("LiabilitiesCurrent", "300000000"),
                     ("CashAndCashEquivalentsAtCarryingValue", "100000000"),
                     ("PropertyPlantAndEquipmentNet", "800000000"),
                     ("StockholdersEquity", "1000000000"),
                     ("LongTermDebtNoncurrent", "400000000"),
                     ("LongTermDebtCurrent", "50000000"),
                     ("CommonStockSharesOutstanding", "50000000")]:
        add_fact(adsh_10k, tag, "0", "20231231", val)

    # --- Q1 2023 10-Q facts (ddate = 20230331, qtrs=1 for flow) ---
    for tag, val in [("Revenues", "240000000"), ("OperatingIncomeLoss", "35000000"),
                     ("NetIncomeLoss", "25000000"),
                     ("NetCashProvidedByUsedInOperatingActivities", "28000000"),
                     ("PaymentsToAcquirePropertyPlantAndEquipment", "7000000")]:
        add_fact(adsh_q1_23, tag, "1", "20230331", val)
    # Balance sheet for Q1 2023
    for tag, val in [("Assets", "1900000000"), ("AssetsCurrent", "480000000"),
                     ("LiabilitiesCurrent", "290000000"),
                     ("CashAndCashEquivalentsAtCarryingValue", "90000000"),
                     ("PropertyPlantAndEquipmentNet", "780000000"),
                     ("StockholdersEquity", "950000000"),
                     ("LongTermDebtNoncurrent", "390000000"),
                     ("LongTermDebtCurrent", "40000000"),
                     ("CommonStockSharesOutstanding", "50000000")]:
        add_fact(adsh_q1_23, tag, "0", "20230331", val)

    # --- Q1 2024 10-Q facts (ddate = 20240331, qtrs=1 for flow) ---
    for tag, val in [("Revenues", "260000000"), ("OperatingIncomeLoss", "40000000"),
                     ("NetIncomeLoss", "30000000"),
                     ("NetCashProvidedByUsedInOperatingActivities", "32000000"),
                     ("PaymentsToAcquirePropertyPlantAndEquipment", "8000000")]:
        add_fact(adsh_q1_24, tag, "1", "20240331", val)
    # Balance sheet for Q1 2024 (MOST RECENT -- these should win for point-in-time)
    for tag, val in [("Assets", "2100000000"), ("AssetsCurrent", "520000000"),
                     ("LiabilitiesCurrent", "310000000"),
                     ("CashAndCashEquivalentsAtCarryingValue", "110000000"),
                     ("PropertyPlantAndEquipmentNet", "820000000"),
                     ("StockholdersEquity", "1050000000"),
                     ("LongTermDebtNoncurrent", "420000000"),
                     ("LongTermDebtCurrent", "45000000"),
                     ("CommonStockSharesOutstanding", "50000000")]:
        add_fact(adsh_q1_24, tag, "0", "20240331", val)

    write_num_txt(base_dir / "num.txt", num_rows)
    return base_dir


@pytest.fixture
def ttm_data_dir(tmp_path: Path) -> Path:
    """Synthetic SEC dir with 10-K + Q1 prior year + Q1 current year."""
    return Path(make_ttm_company_data(tmp_path))


def make_10k_only_data(base_dir: Path) -> str:
    """Company with only FY2023 10-K, no later 10-Q."""
    adsh = "0000012345-24-000001"
    write_sub_txt(base_dir / "sub.txt", [
        {"adsh": adsh, "cik": "12345", "name": "TEST CORP", "sic": "7370",
         "form": "10-K", "period": "20231231", "fy": "2023", "fp": "FY",
         "filed": "20240215"},
    ])
    num_rows = []
    def add(tag, qtrs, val):
        num_rows.append({"adsh": adsh, "tag": tag, "version": "us-gaap/2023",
                         "coreg": "", "ddate": "20231231", "qtrs": qtrs,
                         "uom": "USD", "value": val, "footnote": ""})
    for tag, val in [("Revenues", "1000000000"), ("OperatingIncomeLoss", "150000000"),
                     ("NetIncomeLoss", "100000000"), ("InterestExpense", "5000000")]:
        add(tag, "4", val)
    for tag, val in [("Assets", "2000000000"), ("AssetsCurrent", "500000000"),
                     ("LiabilitiesCurrent", "300000000"),
                     ("CashAndCashEquivalentsAtCarryingValue", "100000000"),
                     ("PropertyPlantAndEquipmentNet", "800000000"),
                     ("StockholdersEquity", "1000000000"),
                     ("LongTermDebtNoncurrent", "400000000"),
                     ("LongTermDebtCurrent", "50000000"),
                     ("CommonStockSharesOutstanding", "50000000")]:
        add(tag, "0", val)
    write_num_txt(base_dir / "num.txt", num_rows)
    return base_dir


@pytest.fixture
def ten_k_only_dir(tmp_path: Path) -> Path:
    """Synthetic SEC dir with only a 10-K, no later 10-Q."""
    return Path(make_10k_only_data(tmp_path))


def make_incomplete_history_data(base_dir: Path) -> str:
    """10-K + current Q1 10-Q, but NO matching prior-year Q1 10-Q."""
    adsh_10k = "0000012345-24-000001"
    adsh_q1 = "0000012345-24-000003"

    write_sub_txt(base_dir / "sub.txt", [
        {"adsh": adsh_10k, "cik": "12345", "name": "TEST CORP", "sic": "7370",
         "form": "10-K", "period": "20231231", "fy": "2023", "fp": "FY",
         "filed": "20240215"},
        {"adsh": adsh_q1, "cik": "12345", "name": "TEST CORP", "sic": "7370",
         "form": "10-Q", "period": "20240331", "fy": "2024", "fp": "Q1",
         "filed": "20240501"},
    ])
    num_rows = []
    def add(adsh, tag, qtrs, ddate, val):
        num_rows.append({"adsh": adsh, "tag": tag, "version": "us-gaap/2023",
                         "coreg": "", "ddate": ddate, "qtrs": qtrs,
                         "uom": "USD", "value": val, "footnote": ""})
    # 10-K
    for tag, val in [("Revenues", "1000000000"), ("OperatingIncomeLoss", "150000000"),
                     ("NetIncomeLoss", "100000000")]:
        add(adsh_10k, tag, "4", "20231231", val)
    for tag, val in [("Assets", "2000000000"), ("AssetsCurrent", "500000000"),
                     ("LiabilitiesCurrent", "300000000"),
                     ("CashAndCashEquivalentsAtCarryingValue", "100000000"),
                     ("PropertyPlantAndEquipmentNet", "800000000"),
                     ("StockholdersEquity", "1000000000"),
                     ("LongTermDebtNoncurrent", "400000000"),
                     ("LongTermDebtCurrent", "50000000")]:
        add(adsh_10k, tag, "0", "20231231", val)
    # Q1 2024
    for tag, val in [("Revenues", "260000000"), ("OperatingIncomeLoss", "40000000"),
                     ("NetIncomeLoss", "30000000")]:
        add(adsh_q1, tag, "1", "20240331", val)
    for tag, val in [("Assets", "2100000000"), ("AssetsCurrent", "520000000"),
                     ("LiabilitiesCurrent", "310000000"),
                     ("CashAndCashEquivalentsAtCarryingValue", "110000000"),
                     ("PropertyPlantAndEquipmentNet", "820000000"),
                     ("StockholdersEquity", "1050000000"),
                     ("LongTermDebtNoncurrent", "420000000"),
                     ("LongTermDebtCurrent", "45000000")]:
        add(adsh_q1, tag, "0", "20240331", val)
    write_num_txt(base_dir / "num.txt", num_rows)
    return base_dir


@pytest.fixture
def incomplete_history_dir(tmp_path: Path) -> Path:
    """10-K + Q1 2024 but no Q1 2023 (prior year)."""
    return Path(make_incomplete_history_data(tmp_path))


def make_ddate_data(base_dir: Path) -> str:
    """Single filing where num.txt has both current (20231231) and prior-year
    (20221231) values for LongTermDebtNoncurrent. The prior-year value is
    LARGER (380) than the current (200) -- a naive MAX(value) aggregation
    would silently pick the wrong one."""
    adsh = "0000012345-24-000001"
    write_sub_txt(base_dir / "sub.txt", [
        {"adsh": adsh, "cik": "12345", "name": "TEST CORP", "sic": "7370",
         "form": "10-K", "period": "20231231", "fy": "2023", "fp": "FY",
         "filed": "20240215"},
    ])
    num_rows = [
        # Current period (20231231) -- THIS is the correct value (200)
        {"adsh": adsh, "tag": "LongTermDebtNoncurrent", "version": "us-gaap/2023",
         "coreg": "", "ddate": "20231231", "qtrs": "0", "uom": "USD",
         "value": "200000000", "footnote": ""},
        # Prior-year comparative (20221231) -- LARGER (380), wrong if picked
        {"adsh": adsh, "tag": "LongTermDebtNoncurrent", "version": "us-gaap/2023",
         "coreg": "", "ddate": "20221231", "qtrs": "0", "uom": "USD",
         "value": "380000000", "footnote": ""},
        # Flow field for the 10-K
        {"adsh": adsh, "tag": "Revenues", "version": "us-gaap/2023",
         "coreg": "", "ddate": "20231231", "qtrs": "4", "uom": "USD",
         "value": "1000000000", "footnote": ""},
        {"adsh": adsh, "tag": "OperatingIncomeLoss", "version": "us-gaap/2023",
         "coreg": "", "ddate": "20231231", "qtrs": "4", "uom": "USD",
         "value": "150000000", "footnote": ""},
        {"adsh": adsh, "tag": "NetIncomeLoss", "version": "us-gaap/2023",
         "coreg": "", "ddate": "20231231", "qtrs": "4", "uom": "USD",
         "value": "100000000", "footnote": ""},
        # Balance sheet
        {"adsh": adsh, "tag": "Assets", "version": "us-gaap/2023",
         "coreg": "", "ddate": "20231231", "qtrs": "0", "uom": "USD",
         "value": "2000000000", "footnote": ""},
        {"adsh": adsh, "tag": "AssetsCurrent", "version": "us-gaap/2023",
         "coreg": "", "ddate": "20231231", "qtrs": "0", "uom": "USD",
         "value": "500000000", "footnote": ""},
        {"adsh": adsh, "tag": "LiabilitiesCurrent", "version": "us-gaap/2023",
         "coreg": "", "ddate": "20231231", "qtrs": "0", "uom": "USD",
         "value": "300000000", "footnote": ""},
        {"adsh": adsh, "tag": "CashAndCashEquivalentsAtCarryingValue", "version": "us-gaap/2023",
         "coreg": "", "ddate": "20231231", "qtrs": "0", "uom": "USD",
         "value": "100000000", "footnote": ""},
        {"adsh": adsh, "tag": "PropertyPlantAndEquipmentNet", "version": "us-gaap/2023",
         "coreg": "", "ddate": "20231231", "qtrs": "0", "uom": "USD",
         "value": "800000000", "footnote": ""},
        {"adsh": adsh, "tag": "StockholdersEquity", "version": "us-gaap/2023",
         "coreg": "", "ddate": "20231231", "qtrs": "0", "uom": "USD",
         "value": "1000000000", "footnote": ""},
        {"adsh": adsh, "tag": "LongTermDebtCurrent", "version": "us-gaap/2023",
         "coreg": "", "ddate": "20231231", "qtrs": "0", "uom": "USD",
         "value": "50000000", "footnote": ""},
        {"adsh": adsh, "tag": "CommonStockSharesOutstanding", "version": "us-gaap/2023",
         "coreg": "", "ddate": "20231231", "qtrs": "0", "uom": "USD",
         "value": "50000000", "footnote": ""},
    ]
    write_num_txt(base_dir / "num.txt", num_rows)
    return base_dir


@pytest.fixture
def ddate_dir(tmp_path: Path) -> Path:
    """Single filing with both current and prior-year ddate for the same tag."""
    return Path(make_ddate_data(tmp_path))


def make_coreg_data(base_dir: Path) -> str:
    """Filing where OperatingIncomeLoss has both a consolidated (coreg=NULL)
    row (value=150) and a subsidiary (coreg=SUBSIDIARY-A) row (value=30).
    The pivot should prefer the consolidated (NULL coreg) row, yielding 150."""
    adsh = "0000012345-24-000001"
    write_sub_txt(base_dir / "sub.txt", [
        {"adsh": adsh, "cik": "12345", "name": "TEST CORP", "sic": "7370",
         "form": "10-K", "period": "20231231", "fy": "2023", "fp": "FY",
         "filed": "20240215"},
    ])
    num_rows = [
        {"adsh": adsh, "tag": "OperatingIncomeLoss", "version": "us-gaap/2023",
         "coreg": "", "ddate": "20231231", "qtrs": "4", "uom": "USD",
         "value": "150000000", "footnote": ""},
        {"adsh": adsh, "tag": "OperatingIncomeLoss", "version": "us-gaap/2023",
         "coreg": "SUBSIDIARY-A", "ddate": "20231231", "qtrs": "4", "uom": "USD",
         "value": "30000000", "footnote": ""},
        # Other fields needed for ratios
        {"adsh": adsh, "tag": "Revenues", "version": "us-gaap/2023",
         "coreg": "", "ddate": "20231231", "qtrs": "4", "uom": "USD",
         "value": "1000000000", "footnote": ""},
        {"adsh": adsh, "tag": "NetIncomeLoss", "version": "us-gaap/2023",
         "coreg": "", "ddate": "20231231", "qtrs": "4", "uom": "USD",
         "value": "100000000", "footnote": ""},
        {"adsh": adsh, "tag": "Assets", "version": "us-gaap/2023",
         "coreg": "", "ddate": "20231231", "qtrs": "0", "uom": "USD",
         "value": "2000000000", "footnote": ""},
        {"adsh": adsh, "tag": "AssetsCurrent", "version": "us-gaap/2023",
         "coreg": "", "ddate": "20231231", "qtrs": "0", "uom": "USD",
         "value": "500000000", "footnote": ""},
        {"adsh": adsh, "tag": "LiabilitiesCurrent", "version": "us-gaap/2023",
         "coreg": "", "ddate": "20231231", "qtrs": "0", "uom": "USD",
         "value": "300000000", "footnote": ""},
        {"adsh": adsh, "tag": "CashAndCashEquivalentsAtCarryingValue", "version": "us-gaap/2023",
         "coreg": "", "ddate": "20231231", "qtrs": "0", "uom": "USD",
         "value": "100000000", "footnote": ""},
        {"adsh": adsh, "tag": "PropertyPlantAndEquipmentNet", "version": "us-gaap/2023",
         "coreg": "", "ddate": "20231231", "qtrs": "0", "uom": "USD",
         "value": "800000000", "footnote": ""},
        {"adsh": adsh, "tag": "StockholdersEquity", "version": "us-gaap/2023",
         "coreg": "", "ddate": "20231231", "qtrs": "0", "uom": "USD",
         "value": "1000000000", "footnote": ""},
        {"adsh": adsh, "tag": "LongTermDebtNoncurrent", "version": "us-gaap/2023",
         "coreg": "", "ddate": "20231231", "qtrs": "0", "uom": "USD",
         "value": "400000000", "footnote": ""},
        {"adsh": adsh, "tag": "LongTermDebtCurrent", "version": "us-gaap/2023",
         "coreg": "", "ddate": "20231231", "qtrs": "0", "uom": "USD",
         "value": "50000000", "footnote": ""},
        {"adsh": adsh, "tag": "CommonStockSharesOutstanding", "version": "us-gaap/2023",
         "coreg": "", "ddate": "20231231", "qtrs": "0", "uom": "USD",
         "value": "50000000", "footnote": ""},
    ]
    write_num_txt(base_dir / "num.txt", num_rows)
    return base_dir


@pytest.fixture
def coreg_dir(tmp_path: Path) -> Path:
    """Filing with consolidated vs subsidiary coreg rows."""
    return Path(make_coreg_data(tmp_path))


def make_tag_alias_data(base_dir: Path) -> str:
    """Filing that uses LongTermDebtAndCapitalLeaseObligations (an alias in
    TAG_MAP for LongTermDebtNoncurrent) instead of the canonical tag name."""
    adsh = "0000012345-24-000001"
    write_sub_txt(base_dir / "sub.txt", [
        {"adsh": adsh, "cik": "12345", "name": "TEST CORP", "sic": "7370",
         "form": "10-K", "period": "20231231", "fy": "2023", "fp": "FY",
         "filed": "20240215"},
    ])
    num_rows = [
        # Uses the ALIAS tag, not the canonical LongTermDebtNoncurrent
        {"adsh": adsh, "tag": "LongTermDebtAndCapitalLeaseObligations",
         "version": "us-gaap/2023", "coreg": "", "ddate": "20231231",
         "qtrs": "0", "uom": "USD", "value": "400000000", "footnote": ""},
        # Standard fields
        {"adsh": adsh, "tag": "Revenues", "version": "us-gaap/2023",
         "coreg": "", "ddate": "20231231", "qtrs": "4", "uom": "USD",
         "value": "1000000000", "footnote": ""},
        {"adsh": adsh, "tag": "OperatingIncomeLoss", "version": "us-gaap/2023",
         "coreg": "", "ddate": "20231231", "qtrs": "4", "uom": "USD",
         "value": "150000000", "footnote": ""},
        {"adsh": adsh, "tag": "NetIncomeLoss", "version": "us-gaap/2023",
         "coreg": "", "ddate": "20231231", "qtrs": "4", "uom": "USD",
         "value": "100000000", "footnote": ""},
        {"adsh": adsh, "tag": "Assets", "version": "us-gaap/2023",
         "coreg": "", "ddate": "20231231", "qtrs": "0", "uom": "USD",
         "value": "2000000000", "footnote": ""},
        {"adsh": adsh, "tag": "AssetsCurrent", "version": "us-gaap/2023",
         "coreg": "", "ddate": "20231231", "qtrs": "0", "uom": "USD",
         "value": "500000000", "footnote": ""},
        {"adsh": adsh, "tag": "LiabilitiesCurrent", "version": "us-gaap/2023",
         "coreg": "", "ddate": "20231231", "qtrs": "0", "uom": "USD",
         "value": "300000000", "footnote": ""},
        {"adsh": adsh, "tag": "CashAndCashEquivalentsAtCarryingValue", "version": "us-gaap/2023",
         "coreg": "", "ddate": "20231231", "qtrs": "0", "uom": "USD",
         "value": "100000000", "footnote": ""},
        {"adsh": adsh, "tag": "PropertyPlantAndEquipmentNet", "version": "us-gaap/2023",
         "coreg": "", "ddate": "20231231", "qtrs": "0", "uom": "USD",
         "value": "800000000", "footnote": ""},
        {"adsh": adsh, "tag": "StockholdersEquity", "version": "us-gaap/2023",
         "coreg": "", "ddate": "20231231", "qtrs": "0", "uom": "USD",
         "value": "1000000000", "footnote": ""},
        {"adsh": adsh, "tag": "LongTermDebtCurrent", "version": "us-gaap/2023",
         "coreg": "", "ddate": "20231231", "qtrs": "0", "uom": "USD",
         "value": "50000000", "footnote": ""},
        {"adsh": adsh, "tag": "CommonStockSharesOutstanding", "version": "us-gaap/2023",
         "coreg": "", "ddate": "20231231", "qtrs": "0", "uom": "USD",
         "value": "50000000", "footnote": ""},
    ]
    write_num_txt(base_dir / "num.txt", num_rows)
    return base_dir


@pytest.fixture
def tag_alias_dir(tmp_path: Path) -> Path:
    """Filing using tag alias (LongTermDebtAndCapitalLeaseObligations)."""
    return Path(make_tag_alias_data(tmp_path))


def make_debt_free_data(base_dir: Path) -> str:
    """Company with no debt whatsoever -- no InterestExpense, LongTermDebt*, or
    ShortTermBorrowings tagged. DebtToEquity should be 0, InterestCoverage inf."""
    adsh = "0000012345-24-000001"
    write_sub_txt(base_dir / "sub.txt", [
        {"adsh": adsh, "cik": "12345", "name": "DEBTFREE INC", "sic": "7370",
         "form": "10-K", "period": "20231231", "fy": "2023", "fp": "FY",
         "filed": "20240215"},
    ])
    num_rows = [
        {"adsh": adsh, "tag": "Revenues", "version": "us-gaap/2023",
         "coreg": "", "ddate": "20231231", "qtrs": "4", "uom": "USD",
         "value": "1000000000", "footnote": ""},
        {"adsh": adsh, "tag": "OperatingIncomeLoss", "version": "us-gaap/2023",
         "coreg": "", "ddate": "20231231", "qtrs": "4", "uom": "USD",
         "value": "200000000", "footnote": ""},
        {"adsh": adsh, "tag": "NetIncomeLoss", "version": "us-gaap/2023",
         "coreg": "", "ddate": "20231231", "qtrs": "4", "uom": "USD",
         "value": "150000000", "footnote": ""},
        {"adsh": adsh, "tag": "Assets", "version": "us-gaap/2023",
         "coreg": "", "ddate": "20231231", "qtrs": "0", "uom": "USD",
         "value": "2000000000", "footnote": ""},
        {"adsh": adsh, "tag": "AssetsCurrent", "version": "us-gaap/2023",
         "coreg": "", "ddate": "20231231", "qtrs": "0", "uom": "USD",
         "value": "500000000", "footnote": ""},
        {"adsh": adsh, "tag": "LiabilitiesCurrent", "version": "us-gaap/2023",
         "coreg": "", "ddate": "20231231", "qtrs": "0", "uom": "USD",
         "value": "300000000", "footnote": ""},
        {"adsh": adsh, "tag": "CashAndCashEquivalentsAtCarryingValue", "version": "us-gaap/2023",
         "coreg": "", "ddate": "20231231", "qtrs": "0", "uom": "USD",
         "value": "500000000", "footnote": ""},
        {"adsh": adsh, "tag": "PropertyPlantAndEquipmentNet", "version": "us-gaap/2023",
         "coreg": "", "ddate": "20231231", "qtrs": "0", "uom": "USD",
         "value": "800000000", "footnote": ""},
        {"adsh": adsh, "tag": "StockholdersEquity", "version": "us-gaap/2023",
         "coreg": "", "ddate": "20231231", "qtrs": "0", "uom": "USD",
         "value": "1500000000", "footnote": ""},
        {"adsh": adsh, "tag": "CommonStockSharesOutstanding", "version": "us-gaap/2023",
         "coreg": "", "ddate": "20231231", "qtrs": "0", "uom": "USD",
         "value": "50000000", "footnote": ""},
        # NOTE: no InterestExpense, LongTermDebt*, or ShortTermBorrowings
    ]
    write_num_txt(base_dir / "num.txt", num_rows)
    return base_dir


@pytest.fixture
def debt_free_dir(tmp_path: Path) -> Path:
    """Company with no debt tags at all."""
    return Path(make_debt_free_data(tmp_path))


# ---------------------------------------------------------------------------
# Fixture: a full pipeline helper for loading synthetic data
# ---------------------------------------------------------------------------

@pytest.fixture
def load_synthetic(ttm_data_dir: Path):
    """Returns (con, history_df) after running the full pipeline on the
    standard TTM synthetic dataset."""
    import core.fundamentals.sec_loader as sec_screen
    con = duckdb.connect()
    sec_screen.load_data_filtered(con, str(ttm_data_dir),
                                  form_types={"10-K", "10-Q"})
    sec_screen.filter_relevant_submissions(con)
    per_filing = sec_screen.build_filing_fundamentals(con)
    sec_screen.accumulate_fundamentals_history(con, per_filing)
    history = sec_screen.load_fundamentals_history(con)
    ttm = sec_screen.compute_ttm(history)
    ratios = sec_screen.compute_ratios(ttm)
    return con, history, ttm, ratios
