"""Ingestion pipeline for SEC Notes Data Set ``txt.txt`` files.

The SEC publishes "Financial Statement and Notes Data Sets" separately from
the regular DERA quarterly data.  These zips contain ``txt.txt`` — a
tab-delimited file with unformatted / HTML text blocks of financial
footnotes (Debt Schedules, Commitments & Contingencies, Going Concern
disclosures, Material Weaknesses, Restatements, etc.).

This module reads ``txt.txt``, filters to high-value footnote tags, joins
with ``sub.txt`` for CIK / fiscal-period metadata, and stores the results
in a persistent DuckDB table ``footnote_text_blocks``.

Unlike the main fundamentals pipeline (which is required for screening),
footnote ingestion is **optional** — the app works fine without it.
"""

import os
from typing import List, Optional, Set

import duckdb
import pandas as pd

from core.fundamentals.data_ingestion import get_db_connection

# ---------------------------------------------------------------------------
# High-value footnote tags — qualitative disclosures that matter for
# fundamental analysis and risk assessment.
# ---------------------------------------------------------------------------
HIGH_VALUE_TAGS: Set[str] = {
    # Liquidity / Going Concern
    "SubstantialDoubtAboutGoingConcernTextBlock",
    "LiquidityVisAVisGoingConcernTextBlock",
    # Debt / Commitments
    "DebtDisclosureTextBlock",
    "CommitmentsAndContingenciesTextBlock",
    "LongTermDebtTextBlock",
    "ScheduleOfDebtTextBlock",
    # Accounting / Restatements
    "RestatementToPriorPeriodBlock",
    "AccountingChangesAndErrorCorrectionsTextBlock",
    # Risk / Legal
    "LegalMattersAndContingenciesTextBlock",
    "RelatedPartyTransactionsTextBlock",
    # Operations / Strategy
    "BasisOfPresentationAndSignificantAccountingPoliciesTextBlock",
    "RevenueFromContractWithCustomerTextBlock",
    "LesseeOperatingLeasesTextBlock",
}


# Human-readable labels for the tag names (shown in the UI).
TAG_LABELS: dict = {
    "SubstantialDoubtAboutGoingConcernTextBlock": "Going Concern",
    "LiquidityVisAVisGoingConcernTextBlock": "Liquidity / Going Concern",
    "DebtDisclosureTextBlock": "Debt Disclosure",
    "CommitmentsAndContingenciesTextBlock": "Commitments & Contingencies",
    "LongTermDebtTextBlock": "Long-Term Debt",
    "ScheduleOfDebtTextBlock": "Debt Schedule",
    "RestatementToPriorPeriodBlock": "Restatement",
    "AccountingChangesAndErrorCorrectionsTextBlock": "Accounting Changes",
    "LegalMattersAndContingenciesTextBlock": "Legal Matters",
    "RelatedPartyTransactionsTextBlock": "Related Party Transactions",
    "BasisOfPresentationAndSignificantAccountingPoliciesTextBlock": "Accounting Policies",
    "RevenueFromContractWithCustomerTextBlock": "Revenue Recognition",
    "LesseeOperatingLeasesTextBlock": "Operating Leases",
}


def ingest_notes_txt(
    con: duckdb.DuckDBPyConnection,
    data_dir: str,
) -> int:
    """Load ``txt.txt`` from *data_dir*, filter to high-value tags, join with
    ``sub.txt`` for CIK/period metadata, and store in ``footnote_text_blocks``.

    Returns the number of text blocks ingested.  If ``txt.txt`` is not
    found, returns 0 (not an error — notes data is optional).
    """
    txt_path = os.path.join(data_dir, "txt.txt")
    if not os.path.exists(txt_path):
        return 0

    # Load the raw text file — it's tab-delimited TSV like the other SEC files.
    con.execute(f"""
        CREATE OR REPLACE TABLE txt_raw AS
        SELECT * FROM read_csv('{txt_path}', delim='\\t', header=True,
                                quote='', all_varchar=True, strict_mode=False);
    """)

    # Filter to our high-value tags only, and keep rows that actually have text.
    tag_list = "', '".join(HIGH_VALUE_TAGS)
    con.execute(f"""
        CREATE OR REPLACE TABLE txt_filtered AS
        SELECT adsh, tag, ddate, TRY_CAST(qtrs AS INTEGER) AS qtrs,
               value AS txt_value
        FROM txt_raw
        WHERE tag IN ('{tag_list}')
          AND value IS NOT NULL
          AND TRIM(value) != '';
    """)

    # Join with sub to get CIK, fiscal year, and fiscal period.
    con.execute("""
        CREATE OR REPLACE TABLE footnote_text_blocks AS
        SELECT t.adsh, s.cik, t.tag, t.ddate, s.fp, TRY_CAST(s.fy AS INTEGER) AS fy,
               t.txt_value
        FROM txt_filtered t
        JOIN sub s ON t.adsh = s.adsh;
    """)

    count = con.execute(
        "SELECT COUNT(*) FROM footnote_text_blocks"
    ).fetchone()[0]

    con.execute("DROP TABLE IF EXISTS txt_raw")
    con.execute("DROP TABLE IF EXISTS txt_filtered")

    if count > 0:
        print(f"[notes] Ingested {count:,} footnote text blocks "
              f"from {os.path.basename(data_dir)}")

    return count


def get_footnotes_for_cik(cik: str) -> pd.DataFrame:
    """Return all footnote text blocks for a given CIK, most recent first."""
    con = get_db_connection()
    try:
        if con.execute(
            "SELECT COUNT(*) FROM information_schema.tables "
            "WHERE table_name = 'footnote_text_blocks'"
        ).fetchone()[0] == 0:
            return pd.DataFrame()

        return con.execute(
            "SELECT * FROM footnote_text_blocks WHERE CAST(cik AS VARCHAR) = ? "
            "ORDER BY ddate DESC, fy DESC",
            [str(cik)],
        ).fetchdf()
    finally:
        con.close()


def has_footnote_data() -> bool:
    """Return True if footnote data has been ingested at least once."""
    con = get_db_connection()
    try:
        return con.execute(
            "SELECT COUNT(*) FROM information_schema.tables "
            "WHERE table_name = 'footnote_text_blocks'"
        ).fetchone()[0] > 0
    finally:
        con.close()
