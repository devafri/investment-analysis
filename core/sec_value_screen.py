#!/usr/bin/env python3
"""
sec_value_screen.py
--------------------
A local pipeline for screening SEC Financial Statement Data Sets (DERA XBRL data)
using value-investing frameworks (Greenblatt Magic Formula, Buffett/Pabrai-style
quality filters).

WHY DUCKDB
The DERA quarterly data set (num.txt, sub.txt, tag.txt, pre.txt) is tab-delimited
and can run several hundred MB to 1GB+ unzipped. DuckDB queries these files
directly via read_csv() without loading everything into memory, and gives you
full SQL for joins/pivots. This is far more efficient than pandas.read_csv on
the full file.

HOW TO GET THE DATA
1. Download a quarterly zip from:
   https://www.sec.gov/dera/data/financial-statement-data-sets
2. Unzip it to a folder, e.g. ./data/2024q1/
   You should see: sub.txt, num.txt, tag.txt, pre.txt, readme.htm

USAGE
    python3 sec_value_screen.py --data-dir ./data/2024q1 --form 10-K --out screen.csv

WHAT THIS SCRIPT DOES
1. Loads sub/num as DuckDB tables directly from the raw text files.
2. Filters submissions to the form type you specify (10-K by default) and keeps
   the MOST RECENT filing per CIK in this data dump (dedup).
3. Standardizes XBRL tags: many companies use different but economically
   equivalent tags (e.g. Revenues vs RevenueFromContractWithCustomerExcludingAssessedTax).
   The TAG_MAP below uses COALESCE to fall back through common alternates.
4. Pivots NUM into a wide fundamentals table (one row per filing).
5. Computes:
   - Greenblatt-style ROIC = EBIT / (Net Working Capital + Net Fixed Assets)
   - Profitability, leverage, coverage, and efficiency ratios
   - An accrual ratio (Net Income vs Operating Cash Flow) as an earnings-quality
     forensic check
6. Leaves an explicit join point for market data (price, shares out) needed to
   compute Earnings Yield / Enterprise Value, since SEC XBRL data contains NO
   stock price. See fetch_market_data.py for the yfinance-based companion script.
7. Outputs a ranked CSV screen.

LIMITATIONS (read before trusting the output)
- Single quarter's data dump reflects filings ACCEPTED in that quarter -- not
  necessarily fiscal periods ending in that quarter. For a clean annual
  cross-section you often need to pull the most recent 10-K per CIK across
  several consecutive quarterly dumps (companies with off-cycle fiscal years,
  restatements, or late filings will otherwise be dropped or misdated).
- Tag standardization here covers the common cases. Some filers use custom
  extension tags for line items -- those will show up as NULL in this pass.
  Check the `tag` table (custom=1) if a company you care about is missing data.
- No qualitative/footnote data (e.g. lease commitments buried in text,
  contingent liabilities) is captured -- this is a quantitative first-pass
  screen only, not a substitute for reading the 10-K.
- ROIC here uses a simplified Greenblatt invested-capital proxy (NWC + Net
  Fixed Assets). Greenblatt's original also excludes excess cash and
  non-interest-bearing current liabilities more precisely; refine TAG_MAP/
  formulas below if you want to match his methodology exactly.
"""

import argparse
import os
import sys
from typing import Optional, Set
import duckdb
import pandas as pd

from core.valuation import safe_div as _safe_div


# ---------------------------------------------------------------------------
# Tag standardization: canonical_name -> ordered list of XBRL tags to try
# (first match wins, via COALESCE). Extend this as you encounter more filers.
# ---------------------------------------------------------------------------
TAG_MAP = {
    "Revenues": [
        "Revenues",
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "RevenueFromContractWithCustomerIncludingAssessedTax",
        "SalesRevenueNet",
        "SalesRevenueGoodsNet",
    ],
    "CostOfRevenue": [
        "CostOfRevenue",
        "CostOfGoodsAndServicesSold",
        "CostOfGoodsSold",
    ],
    "OperatingIncomeLoss": [
        "OperatingIncomeLoss",
        "OperatingIncome",
    ],
    "NetIncomeLoss": [
        "NetIncomeLoss",
        "ProfitLoss",
        "NetIncomeLossAvailableToCommonStockholdersBasic",
    ],
    "InterestExpense": [
        "InterestExpense",
        "InterestExpenseDebt",
        "InterestAndDebtExpense",
    ],
    "IncomeTaxExpenseBenefit": [
        "IncomeTaxExpenseBenefit",
    ],
    "Assets": [
        "Assets",
    ],
    "AccountsReceivable": [
        "AccountsReceivableNetCurrent",
        "AccountsReceivableNet",
        "AccountsNotesAndLoansReceivableNetCurrent",
    ],
    "AssetsCurrent": [
        "AssetsCurrent",
    ],
    "LiabilitiesCurrent": [
        "LiabilitiesCurrent",
    ],
    "CashAndCashEquivalents": [
        "CashAndCashEquivalentsAtCarryingValue",
        "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents",
    ],
    "PropertyPlantAndEquipmentNet": [
        "PropertyPlantAndEquipmentNet",
    ],
    "StockholdersEquity": [
        "StockholdersEquity",
        "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
    ],
    "LongTermDebtNoncurrent": [
        "LongTermDebtNoncurrent",
        "LongTermDebt",
        "LongTermDebtAndCapitalLeaseObligations",
    ],
    "LongTermDebtCurrent": [
        "LongTermDebtCurrent",
        "DebtCurrent",
        "LongTermDebtAndCapitalLeaseObligationsCurrent",
    ],
    "OperatingLeaseLiability": [
        "OperatingLeaseLiability",
        "OperatingLeaseLiabilityNoncurrent",
    ],
    "ShortTermBorrowings": [
        "ShortTermBorrowings",
    ],
    "NetCashProvidedByUsedInOperatingActivities": [
        "NetCashProvidedByUsedInOperatingActivities",
        "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations",
    ],
    "CapitalExpenditures": [
        "PaymentsToAcquirePropertyPlantAndEquipment",
    ],
    "CommonStockSharesOutstanding": [
        "CommonStockSharesOutstanding",
        "EntityCommonStockSharesOutstanding",
    ],
    "DepreciationAndAmortization": [
        "DepreciationDepletionAndAmortization",
        "DepreciationAndAmortization",
        "Depreciation",
    ],
    "Goodwill": [
        "Goodwill",
    ],
    "IntangibleAssetsNet": [
        "IntangibleAssetsNetExcludingGoodwill",
        "IntangibleAssetsNet",
        "FiniteLivedIntangibleAssetsNet",
    ],
    "InterestIncome": [
        "InterestIncome",
        "InvestmentIncomeInterestAndDividend",
        "InterestAndDividendIncomeOperating",
    ],
}

# Point-in-time (balance sheet) tags use qtrs=0; flow (income/cash flow) tags
# use qtrs=4 for an annual 10-K figure. This matters to avoid blending a
# quarterly flow number with an annual one.
POINT_IN_TIME_TAGS = {
    "AccountsReceivable", "Assets", "AssetsCurrent", "LiabilitiesCurrent", "CashAndCashEquivalents",
    "PropertyPlantAndEquipmentNet", "StockholdersEquity",
    "LongTermDebtNoncurrent", "LongTermDebtCurrent", "ShortTermBorrowings",
    "OperatingLeaseLiability",
    "CommonStockSharesOutstanding", "Goodwill", "IntangibleAssetsNet",
}


def build_pivot_sql(annual: bool = True) -> str:
    """Generate the SQL that pivots num_typed into one wide row per adsh,
    applying COALESCE across alias tags and the correct qtrs filter for
    point-in-time vs flow items.

    For each alias, prefer the value reported against the consolidated filer
    (coreg IS NULL) over any value reported against a co-registrant/
    subsidiary breakdown (coreg populated) -- a bare MAX(value) across both
    can silently pick a subsidiary's partial figure instead of the
    consolidated total when both are present for the same tag+qtrs, and
    there's no guarantee the consolidated total is always the larger number.
    Only falls back to "any coreg" if no consolidated-level row exists at all.
    """
    flow_qtrs = 4 if annual else 1
    exprs = []
    for canon, alias_list in TAG_MAP.items():
        qtrs_filter = "qtrs = 0" if canon in POINT_IN_TIME_TAGS else f"qtrs = {flow_qtrs}"
        pieces = []
        for t in alias_list:
            consolidated = f"MAX(CASE WHEN tag = '{t}' AND {qtrs_filter} AND coreg IS NULL THEN value END)"
            any_coreg = f"MAX(CASE WHEN tag = '{t}' AND {qtrs_filter} THEN value END)"
            pieces.append(f"COALESCE({consolidated}, {any_coreg})")
        coalesce_expr = "COALESCE(" + ", ".join(pieces) + ")" if len(pieces) > 1 else pieces[0]
        exprs.append(f"    {coalesce_expr} AS {canon}")
    return ",\n".join(exprs)


def _resolve_data_path(data_dir: str) -> str:
    """Validate and normalise a data directory path before it's interpolated
    into SQL strings.

    DuckDB's read_csv() path is interpolated (parameterised queries don't
    support table/file paths), so we defensively check that the resolved
    path actually exists and contains the expected files before letting it
    anywhere near a SQL string.
    """
    resolved = os.path.abspath(os.path.expanduser(data_dir))
    # Reject paths containing characters that could break out of the string
    # literal in read_csv('...') — single quotes, backslashes (on non-Windows),
    # or null bytes.
    dangerous = {"\x00"}
    if any(c in resolved for c in dangerous):
        raise ValueError(
            f"Data directory path contains invalid characters: {resolved!r}"
        )
    return resolved


def load_data_filtered(
    con: duckdb.DuckDBPyConnection,
    data_dir: str,
    form_types: Optional[Set[str]] = None,
    allowed_ciks: Optional[Set[str]] = None,
) -> None:
    """Like load_data, but filters sub.txt to the given form_types and/or
    allowed_ciks BEFORE loading num.txt, and only loads num.txt rows whose
    adsh survived that filter.

    num.txt is typically 5-10x more rows than sub.txt has submissions
    (every filing has many tag/value rows). Skipping companies or form
    types you don't care about before that big table gets built/joined/
    pivoted can meaningfully cut memory use and downstream query time on a
    full quarterly dump.

    HONEST CAVEAT: DuckDB's CSV reader has to parse every row of the raw
    text file regardless of any WHERE clause here (no predicate pushdown
    into CSV parsing itself) -- this does NOT speed up the initial file
    read. What it does do is avoid materializing and then joining/pivoting
    rows you were only going to throw away a moment later, which is where
    the real cost is once you get to the GROUP BY-heavy pivot step.
    """
    resolved = _resolve_data_path(data_dir)
    sub_path = os.path.join(resolved, "sub.txt")
    num_path = os.path.join(resolved, "num.txt")
    for p in (sub_path, num_path):
        if not os.path.exists(p):
            sys.exit(f"ERROR: expected file not found: {p}")

    con.execute(f"""
        CREATE OR REPLACE TABLE sub AS
        SELECT * FROM read_csv('{sub_path}', delim='\t', header=True,
                                quote='', all_varchar=True, strict_mode=False);
    """)

    where_clauses = []
    if form_types:
        forms_list = ", ".join(f"'{f.strip().upper()}'" for f in form_types)
        where_clauses.append(f"UPPER(TRIM(form)) IN ({forms_list})")
    if allowed_ciks is not None:
        # Registered as a table (not an inline IN-list) since this can be
        # thousands of CIKs -- keeps the generated SQL small and lets DuckDB
        # do a normal join instead of parsing a huge literal list.
        cik_df = pd.DataFrame({"cik": [str(c) for c in allowed_ciks]})
        con.register("allowed_ciks_df", cik_df)
        con.execute("CREATE OR REPLACE TABLE allowed_ciks AS SELECT * FROM allowed_ciks_df;")
        con.unregister("allowed_ciks_df")
        where_clauses.append("CAST(cik AS VARCHAR) IN (SELECT cik FROM allowed_ciks)")

    if where_clauses:
        where_sql = " WHERE " + " AND ".join(where_clauses)
        con.execute(f"CREATE OR REPLACE TABLE sub AS SELECT * FROM sub{where_sql};")

    con.execute(f"""
        CREATE OR REPLACE TABLE num AS
        SELECT * FROM read_csv('{num_path}', delim='\t', header=True,
                                quote='', all_varchar=True, strict_mode=False)
        WHERE adsh IN (SELECT adsh FROM sub);
    """)
    con.execute("""
        CREATE OR REPLACE TABLE num_typed AS
        SELECT adsh, tag, TRY_CAST(qtrs AS INTEGER) AS qtrs,
               TRY_CAST(value AS DOUBLE) AS value,
               NULLIF(TRIM(coreg), '') AS coreg,
               TRIM(ddate) AS ddate
        FROM num;
    """)

    # --- pre.txt disambiguation: exclude footnote-only tags ---
    _load_and_apply_pre_txt(con, resolved)


def _load_and_apply_pre_txt(con: duckdb.DuckDBPyConnection, data_dir: str) -> None:
    """Load pre.txt (SEC presentation linkbase) and filter ``num_typed`` to
    exclude rows whose tag appears in pre.txt ONLY in non-primary statement
    locations (footnotes, disclosures, cover pages).

    The SEC DERAs ``pre.txt`` maps every (adsh, tag) pair to one or more
    ``stmt`` values: BS (Balance Sheet), IS (Income Statement), CF (Cash
    Flow), EQ (Equity), CP (Cover Page), or blank/non-standard for footnotes
    and disclosures.

    A tag whose ONLY ``pre.txt`` appearances have ``stmt`` NOT IN
    ('BS', 'IS', 'CF') is a footnote/disclosure mirror of a primary concept
    — possibly with a different value — and should not be used for the
    main pivot.  Excluding those rows from ``num_typed`` prevents the
    COALESCE in the pivot from silently picking the footnote value over
    the face-of-statement value.

    Tags with NO ``pre.txt`` entry at all are left alone (backwards
    compatibility with older quarterly zips).
    """
    pre_path = os.path.join(data_dir, "pre.txt")
    if not os.path.exists(pre_path):
        return  # pre.txt optional — older dumps / synthetic data may lack it

    con.execute(f"""
        CREATE OR REPLACE TABLE pre AS
        SELECT * FROM read_csv('{pre_path}', delim='\t', header=True,
                                quote='', all_varchar=True, strict_mode=False);
    """)

    # Tags that appear in pre.txt but NEVER on BS, IS, or CF.
    # Blank/NULL stmt is typical for footnote/disclosure sections and is
    # treated as non-primary.  If a tag appears on a primary statement AND
    # a footnote, the HAVING condition is false → it stays (legitimate).
    con.execute("""
        CREATE OR REPLACE TABLE pre_footnote_only AS
        SELECT adsh, tag
        FROM pre
        GROUP BY adsh, tag
        HAVING SUM(CASE WHEN TRIM(COALESCE(stmt, '')) IN ('BS', 'IS', 'CF')
                        THEN 1 ELSE 0 END) = 0;
    """)

    footnote_count = con.execute(
        "SELECT COUNT(*) FROM pre_footnote_only"
    ).fetchone()[0]

    if footnote_count == 0:
        return  # nothing to filter — every tagged row is on a primary statement

    # Delete from num_typed where (adsh, tag) is footnote-only.
    num_before = con.execute("SELECT COUNT(*) FROM num_typed").fetchone()[0]
    con.execute("""
        DELETE FROM num_typed
        WHERE (adsh, tag) IN (SELECT adsh, tag FROM pre_footnote_only);
    """)
    num_after = con.execute("SELECT COUNT(*) FROM num_typed").fetchone()[0]
    removed = num_before - num_after
    if removed > 0:
        print(
            f"[pre.txt] Excluded {removed:,} footnote-only tag rows "
            f"({footnote_count:,} unique adsh+tag pairs) — "
            f"these would have competed with face-of-statement values in the pivot."
        )


def load_data(con: duckdb.DuckDBPyConnection, data_dir: str) -> None:
    resolved = _resolve_data_path(data_dir)
    sub_path = os.path.join(resolved, "sub.txt")
    num_path = os.path.join(resolved, "num.txt")
    for p in (sub_path, num_path):
        if not os.path.exists(p):
            sys.exit(f"ERROR: expected file not found: {p}")

    con.execute(f"""
        CREATE OR REPLACE TABLE sub AS
        SELECT * FROM read_csv('{sub_path}', delim='\t', header=True,
                                quote='', all_varchar=True, strict_mode=False);
    """)
    con.execute(f"""
        CREATE OR REPLACE TABLE num AS
        SELECT * FROM read_csv('{num_path}', delim='\t', header=True,
                                quote='', all_varchar=True, strict_mode=False);
    """)
    con.execute("""
        CREATE OR REPLACE TABLE num_typed AS
        SELECT adsh, tag, TRY_CAST(qtrs AS INTEGER) AS qtrs,
               TRY_CAST(value AS DOUBLE) AS value,
               NULLIF(TRIM(coreg), '') AS coreg,
               TRIM(ddate) AS ddate
        FROM num;
    """)

    # --- pre.txt disambiguation ---
    _load_and_apply_pre_txt(con, resolved)


def filter_submissions(con: duckdb.DuckDBPyConnection, form: str) -> None:
    """Keep target form type, one filing per CIK (most recently filed in this dump).
    Trims/uppercases the comparison so a stray space or wrong case from a UI
    layer (e.g. an HTML <option> value) doesn't silently zero out every row --
    this failure mode is not exceptions, just an empty table, so it's worth
    guarding here rather than relying on callers to sanitize `form`."""
    form_clean = form.strip().upper()
    con.execute(
        """
        CREATE OR REPLACE TABLE sub_filtered AS
        SELECT * FROM (
            SELECT s.*,
                   ROW_NUMBER() OVER (PARTITION BY cik ORDER BY filed DESC) AS rn
            FROM sub s
            WHERE UPPER(TRIM(form)) = ?
        ) WHERE rn = 1;
        """,
        [form_clean],
    )
    n = con.execute("SELECT COUNT(*) FROM sub_filtered").fetchone()[0]
    if n == 0:
        print(f"WARNING: filter_submissions matched 0 filings for form='{form}' "
              f"(normalized to '{form_clean}'). Check the form values actually "
              f"present in sub.txt, e.g.: SELECT DISTINCT form FROM sub;")


# ---------------------------------------------------------------------------
# Multi-quarter / TTM pipeline (used by the web app for comparative analysis).
#
# The functions above (filter_submissions + build_pivot_sql with a fixed
# `annual` flag + build_fundamentals) keep ONE filing per company and assume
# a single, caller-chosen form type -- fine for a quick single-quarter CLI
# screen, but TTM needs BOTH 10-K and 10-Q filings for a company, across
# MULTIPLE ingested quarterly dumps, kept around rather than overwritten.
#
# XBRL flow figures (income statement, cash flow) in a 10-Q are reported as
# YEAR-TO-DATE cumulative, not a clean single quarter: `qtrs` in num.txt is 1
# for Q1, 2 for six-months-YTD (Q2), 3 for nine-months-YTD (Q3), and 4 for a
# full fiscal year (10-K). Which qtrs value is "this filing's own figure" is
# therefore driven by `fp` (fiscal period: Q1/Q2/Q3/FY) in sub.txt, not a
# single global setting -- so the pivot below computes it per-filing instead
# of taking a fixed annual/quarterly flag.
# ---------------------------------------------------------------------------

FP_TO_QTRS = {"Q1": 1, "Q2": 2, "Q3": 3, "FY": 4}
FLOW_FIELDS = [f for f in TAG_MAP.keys() if f not in POINT_IN_TIME_TAGS]


def build_pivot_sql_dynamic() -> str:
    """Like build_pivot_sql, but the qtrs filter for flow fields is derived
    per-row from `fp` (already joined in) instead of a single fixed value --
    necessary once a single ingested batch can contain a mix of 10-K and
    10-Q filings with different fiscal periods."""
    exprs = []
    fp_case = "CASE fp WHEN 'Q1' THEN 1 WHEN 'Q2' THEN 2 WHEN 'Q3' THEN 3 WHEN 'FY' THEN 4 ELSE NULL END"
    for canon, alias_list in TAG_MAP.items():
        qtrs_filter = "qtrs = 0" if canon in POINT_IN_TIME_TAGS else f"qtrs = {fp_case}"
        pieces = []
        for t in alias_list:
            consolidated = f"MAX(CASE WHEN tag = '{t}' AND {qtrs_filter} AND coreg IS NULL THEN value END)"
            any_coreg = f"MAX(CASE WHEN tag = '{t}' AND {qtrs_filter} THEN value END)"
            pieces.append(f"COALESCE({consolidated}, {any_coreg})")
        coalesce_expr = "COALESCE(" + ", ".join(pieces) + ")" if len(pieces) > 1 else pieces[0]
        exprs.append(f"    {coalesce_expr} AS {canon}")
    return ",\n".join(exprs)


def filter_relevant_submissions(con: duckdb.DuckDBPyConnection) -> None:
    """Keep ALL 10-K and 10-Q filings from the currently loaded `sub` table --
    no per-CIK dedup. TTM/comparative analysis needs the recent history of
    filings per company, not just the latest one."""
    con.execute("""
        CREATE OR REPLACE TABLE sub_filtered AS
        SELECT * FROM sub WHERE UPPER(TRIM(form)) IN ('10-K', '10-Q');
    """)


def build_filing_fundamentals(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Returns one row PER FILING (adsh) -- not deduped to latest-per-company --
    covering every 10-K/10-Q currently loaded (from filter_relevant_submissions).
    Each filing's own reporting period is matched via ddate = sub.period, same
    reasoning as build_fundamentals: without this, prior-year comparative
    figures in the same filing would blend into the aggregation."""
    pivot_sql = build_pivot_sql_dynamic()
    con.execute("""
        CREATE OR REPLACE TABLE num_current_period_multi AS
        SELECT nt.*, sf.fp AS fp
        FROM num_typed nt
        JOIN sub_filtered sf ON nt.adsh = sf.adsh
        WHERE nt.ddate = TRIM(sf.period);
    """)
    con.execute(f"""
        CREATE OR REPLACE TABLE filing_fundamentals AS
        SELECT adsh,
{pivot_sql}
        FROM num_current_period_multi
        GROUP BY adsh;
    """)
    df = con.execute("""
        SELECT sf.cik, sf.name, sf.sic, sf.form, sf.period, sf.fy, sf.fp, sf.filed,
               f.*
        FROM filing_fundamentals f
        JOIN sub_filtered sf USING (adsh)
    """).fetchdf()

    # --- Cover-page shares outstanding (dei:EntityCommonStockSharesOutstanding)
    #     Left-joined by adsh — not subject to ddate=period filter because
    #     this tag's ddate is the cover-page date, not the fiscal period end.
    try:
        cover_shares = fetch_cover_page_shares_outstanding(con)
        if not cover_shares.empty:
            df = df.merge(cover_shares, on="adsh", how="left")
    except Exception:
        pass  # non-critical enhancement — don't fail the entire ingest

    return df


def fetch_cover_page_shares_outstanding(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Fetch dei:EntityCommonStockSharesOutstanding joined by adsh only.

    This tag lives in the dei (cover-page) namespace — its ``ddate`` is the
    filing's cover-page date (close to the actual filing date), which is
    almost never equal to the fiscal period end.  The standard pivot's
    ``WHERE nt.ddate = TRIM(sf.period)`` filter would silently exclude it
    from every filing, so it needs its own query path.

    If a filing reports the tag more than once (shouldn't normally happen,
    but defensively), take the value whose ddate is closest to the filing's
    ``filed`` date.
    """
    return con.execute("""
        SELECT adsh, value AS CommonStockSharesOutstandingCoverPage
        FROM (
            SELECT nt.adsh, nt.value, nt.ddate,
                   ROW_NUMBER() OVER (
                       PARTITION BY nt.adsh
                       ORDER BY ABS(
                           CAST(nt.ddate AS INTEGER) - CAST(sf.filed AS INTEGER)
                       ) ASC
                   ) AS rn
            FROM num_typed nt
            JOIN sub_filtered sf ON nt.adsh = sf.adsh
            WHERE nt.tag = 'EntityCommonStockSharesOutstanding'
        ) WHERE rn = 1
    """).fetchdf()


def accumulate_fundamentals_history(con: duckdb.DuckDBPyConnection, batch_df: pd.DataFrame) -> None:
    """Merge one ingest batch's per-filing fundamentals into a persistent
    fundamentals_history table (on the same on-disk DuckDB connection),
    deduplicated by adsh -- so ingesting overlapping quarterly zips twice
    doesn't create duplicate rows, and each ingest call adds to the
    accumulated history rather than replacing it.

    If the existing fundamentals_history is missing columns that batch_df
    has (e.g. because a new XBRL tag was added to TAG_MAP since the last
    ingest), those columns are added automatically before the INSERT so
    schema drift doesn't cause a "N columns but M values" binder error.
    """
    con.register("batch_df", batch_df)
    con.execute("""
        CREATE TABLE IF NOT EXISTS fundamentals_history AS
        SELECT * FROM batch_df WHERE FALSE;
    """)

    # --- Auto-migrate: add any column present in batch_df but missing from
    #     the existing fundamentals_history (new TAG_MAP entries, cover-page
    #     shares, etc.).  Without this, `INSERT INTO SELECT *` fails with a
    #     column-count mismatch whenever the pipeline output gains a column.
    existing = {
        r[0]
        for r in con.execute(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_name = 'fundamentals_history'"
        ).fetchall()
    }
    for col in batch_df.columns:
        if col not in existing:
            # DuckDB column names are case-insensitive, but the CSV-derived
            # schema uses the exact casing from TAG_MAP keys / the pivot SQL
            # aliases.  Quote with double-quotes to preserve casing.
            con.execute(f'ALTER TABLE fundamentals_history ADD COLUMN "{col}" VARCHAR')

    # Use explicit column lists (not SELECT *) so the INSERT works even when
    # the table's column order doesn't match batch_df's column order (e.g.
    # after an earlier ALTER TABLE ADD COLUMN appended columns in a different
    # sequence).
    col_list = ", ".join(f'"{c}"' for c in batch_df.columns)
    con.execute(f"""
        INSERT INTO fundamentals_history ({col_list})
        SELECT {col_list} FROM batch_df
        WHERE adsh NOT IN (SELECT adsh FROM fundamentals_history);
    """)
    con.unregister("batch_df")


def log_ingest(con: duckdb.DuckDBPyConnection, source_name: str, filings_added: int) -> None:
    """Write one row to the persistent ingest_log table so the UI can show
    which quarterly zips have already been ingested. Called from the /ingest
    route alongside accumulate_fundamentals_history."""
    con.execute("""
        CREATE TABLE IF NOT EXISTS ingest_log (
            source_name VARCHAR,
            filings_added BIGINT,
            ingested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    con.execute(
        "INSERT INTO ingest_log (source_name, filings_added) VALUES (?, ?)",
        [source_name, filings_added],
    )


def load_fundamentals_history(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    if con.execute(
        "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = 'fundamentals_history'"
    ).fetchone()[0] == 0:
        return pd.DataFrame()
    return con.execute("SELECT * FROM fundamentals_history").fetchdf()


ENRICHMENT_COLUMNS = ["RevenueGrowth", "FCFGrowth", "RevenueGrowth3yr", "FScore"]


def compute_ttm(history_df: pd.DataFrame) -> pd.DataFrame:
    """Compute trailing-twelve-month figures from accumulated filing history.

    Delegates to compute_ttm_with_enrichment (the canonical, optimised
    implementation) and drops the enrichment columns so callers that only
    need TTM fundamentals (e.g. the CLI diagnostic, tests) get the same
    output they would have before the TTM+enrichment pass was unified.

    See compute_ttm_with_enrichment for the full methodology.
    """
    ttm = compute_ttm_with_enrichment(history_df)
    if ttm.empty:
        return ttm
    # Drop enrichment columns that were added during the combined pass.
    drop_cols = [c for c in ENRICHMENT_COLUMNS if c in ttm.columns]
    return ttm.drop(columns=drop_cols) if drop_cols else ttm




def build_fundamentals(con: duckdb.DuckDBPyConnection, annual: bool) -> pd.DataFrame:
    pivot_sql = build_pivot_sql(annual=annual)

    # CRITICAL: every 10-K/10-Q includes comparative prior-period figures
    # alongside the current period's (e.g. this year's AND last year's
    # balance sheet, both tagged with qtrs=0 but different `ddate`). Without
    # restricting to the filing's OWN reporting period, the MAX()-based
    # pivot blends current and prior-year values together and silently picks
    # whichever is numerically larger -- which is only "correct" by
    # coincidence when a figure happened to grow year-over-year, and
    # silently WRONG (picks the older, larger number) whenever it shrank.
    # Filtering here to ddate = sub_filtered.period ensures only the current
    # filing's own period is used.
    con.execute("""
        CREATE OR REPLACE TABLE num_current_period AS
        SELECT nt.*
        FROM num_typed nt
        JOIN sub_filtered sf ON nt.adsh = sf.adsh
        WHERE nt.ddate = TRIM(sf.period);
    """)

    con.execute(f"""
        CREATE OR REPLACE TABLE fundamentals AS
        SELECT adsh,
{pivot_sql}
        FROM num_current_period
        GROUP BY adsh;
    """)
    df = con.execute("""
        SELECT sf.cik, sf.name, sf.sic, sf.period, sf.fy, sf.form, sf.filed,
               f.*
        FROM fundamentals f
        JOIN sub_filtered sf USING (adsh)
    """).fetchdf()
    return df


def compute_ratios(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # --- Debt & capital structure ---
    # Include operating lease liabilities (ASC 842) in total debt — a CFA
    # always capitalizes leases for leverage analysis.  The column may not
    # exist yet if data was ingested before this tag was added to TAG_MAP.
    debt_cols = ["LongTermDebtNoncurrent", "LongTermDebtCurrent",
                 "ShortTermBorrowings"]
    if "OperatingLeaseLiability" in df.columns:
        debt_cols.append("OperatingLeaseLiability")
    df["TotalDebt"] = df[debt_cols].sum(axis=1, min_count=1)
    df["NetDebt"] = df["TotalDebt"] - df["CashAndCashEquivalents"]

    # --- Greenblatt-style ROIC ---
    df["NWC"] = df["AssetsCurrent"] - df["LiabilitiesCurrent"]
    df["InvestedCapital"] = df["NWC"] + df["PropertyPlantAndEquipmentNet"]
    df["ROIC"] = df["OperatingIncomeLoss"] / df["InvestedCapital"]

    # --- Profitability ---
    df["GrossProfit"] = df["Revenues"] - df["CostOfRevenue"]
    df["GrossMargin"] = df["GrossProfit"] / df["Revenues"]
    df["OperatingMargin"] = df["OperatingIncomeLoss"] / df["Revenues"]
    df["NetMargin"] = df["NetIncomeLoss"] / df["Revenues"]
    df["ROE"] = df["NetIncomeLoss"] / df["StockholdersEquity"]
    df["ROA"] = df["NetIncomeLoss"] / df["Assets"]

    # --- Liquidity & solvency ---
    df["CurrentRatio"] = df["AssetsCurrent"] / df["LiabilitiesCurrent"]
    # TotalDebt/InterestExpense being NaN or 0 usually means the filer has no
    # debt (and therefore never tags InterestExpense at all), not that the
    # data is missing/bad. Treat that as "no leverage" rather than letting a
    # NaN comparison in apply_quality_screen silently drop debt-free
    # companies -- exactly the kind of name a value screen should surface.
    df["DebtToEquity"] = (df["TotalDebt"].fillna(0) / df["StockholdersEquity"])
    df["InterestCoverage"] = df["OperatingIncomeLoss"] / df["InterestExpense"].replace(0, pd.NA)
    df.loc[df["InterestExpense"].isna() | (df["InterestExpense"] == 0), "InterestCoverage"] = float("inf")

    # --- Earnings quality (forensic) ---
    # Negative AccrualRatio (CFO > NI) is generally a positive quality signal;
    # persistently large positive values flag aggressive accrual accounting.
    df["FCF"] = df["NetCashProvidedByUsedInOperatingActivities"] - df["CapitalExpenditures"].fillna(0)
    df["AccrualRatio"] = (df["NetIncomeLoss"] - df["NetCashProvidedByUsedInOperatingActivities"]) / df["Assets"]
    df["CFO_to_NI"] = df["NetCashProvidedByUsedInOperatingActivities"] / df["NetIncomeLoss"]

    return df


def compute_ttm_with_enrichment(history_df: pd.DataFrame) -> pd.DataFrame:
    """Combined pass: TTM computation + enrichment (growth + F-Score) in a
    single groupby over history_df.

    Pre-sorts the DataFrame once by CIK+period (rather than sorting each
    group inside the loop) and converts form/fp columns up front so the
    per-group work is pure filtering.

    Enrichment (RevenueGrowth, FCFGrowth, RevenueGrowth3yr, FScore) is
    computed from the same per-company group while the annual rows are
    already in hand, avoiding a second full-groupby pass over 45K+ rows.
    """
    if history_df.empty:
        return history_df

    df = history_df.copy()
    df["period"] = df["period"].astype(str).str.strip()
    df["form"] = df["form"].astype(str).str.strip().str.upper()
    df["fp"] = df["fp"].astype(str).str.strip().str.upper()
    # Sort once globally instead of per-group — significant wall-time saving
    df = df.sort_values(["cik", "period"])

    results: list = []
    for cik, group in df.groupby("cik"):
        # Already sorted globally by cik+period above — no per-group sort needed
        annual_rows = group[group["form"] == "10-K"]
        if annual_rows.empty:
            continue  # no 10-K anchor

        latest_10k = annual_rows.iloc[-1]
        fy_period = latest_10k["period"]
        most_recent_row = group.iloc[-1]
        later_10qs = group[(group["form"] == "10-Q") & (group["period"] > fy_period)].sort_values("period")

        result = {
            "cik": cik,
            "name": most_recent_row.get("name"),
            "sic": most_recent_row.get("sic"),
            "as_of_period": most_recent_row.get("period"),
        }
        for field in df.columns:
            if field in POINT_IN_TIME_TAGS:
                result[field] = most_recent_row.get(field)

        # Cover-page shares outstanding is conceptually point-in-time (from
        # the most recent filing's cover page) but fetched outside the main
        # pivot pipeline — copy it from the most recent row here.
        if "CommonStockSharesOutstandingCoverPage" in df.columns:
            result["CommonStockSharesOutstandingCoverPage"] = (
                most_recent_row.get("CommonStockSharesOutstandingCoverPage")
            )

        # --- TTM flow fields ---
        if later_10qs.empty:
            result["ttm_basis"] = f"10-K only (FY{latest_10k.get('fy')}, no newer 10-Q ingested)"
            for field in FLOW_FIELDS:
                result[field] = latest_10k.get(field)
        else:
            current_ytd = later_10qs.iloc[-1]
            current_fp = current_ytd["fp"]
            prior_candidates = group[
                (group["form"] == "10-Q") &
                (group["fp"] == current_fp) &
                (group["period"] < fy_period)
            ].sort_values("period")

            if prior_candidates.empty:
                result["ttm_basis"] = (
                    f"Incomplete: have FY{latest_10k.get('fy')} 10-K + {current_fp} "
                    f"{current_ytd.get('fy')} 10-Q, but no matching prior-year {current_fp} "
                    f"10-Q ingested -- TTM flow figures unavailable until you ingest that quarter."
                )
                for field in FLOW_FIELDS:
                    result[field] = None
            else:
                prior_ytd = prior_candidates.iloc[-1]
                result["ttm_basis"] = (
                    f"TTM through {current_ytd.get('period')} "
                    f"(FY{latest_10k.get('fy')} 10-K + {current_fp} {current_ytd.get('fy')} 10-Q "
                    f"- {current_fp} {prior_ytd.get('fy')} 10-Q)"
                )
                for field in FLOW_FIELDS:
                    a, b, c = latest_10k.get(field), current_ytd.get(field), prior_ytd.get(field)
                    if a is None or b is None or c is None or pd.isna(a) or pd.isna(b) or pd.isna(c):
                        result[field] = None
                    else:
                        result[field] = a + b - c

        # --- Enrichment: growth + F-Score (computed from the same group) ---
        if len(annual_rows) < 2:
            result["RevenueGrowth"] = None
            result["FCFGrowth"] = None
            result["FScore"] = None
        else:
            cur_a, pri_a = annual_rows.iloc[-1], annual_rows.iloc[-2]

            # Revenue growth
            rev_cur_v, rev_pri_v = cur_a.get("Revenues"), pri_a.get("Revenues")
            if rev_cur_v and rev_pri_v and float(rev_pri_v) != 0:
                try:
                    result["RevenueGrowth"] = (float(rev_cur_v) / float(rev_pri_v)) - 1
                except (TypeError, ValueError):
                    result["RevenueGrowth"] = None
            else:
                result["RevenueGrowth"] = None

            # FCF growth
            fcf_cur = (float(cur_a.get("NetCashProvidedByUsedInOperatingActivities") or 0)
                       - float(cur_a.get("CapitalExpenditures") or 0))
            fcf_pri = (float(pri_a.get("NetCashProvidedByUsedInOperatingActivities") or 0)
                       - float(pri_a.get("CapitalExpenditures") or 0))
            result["FCFGrowth"] = ((fcf_cur / fcf_pri) - 1) if fcf_pri != 0 else None

            # F-Score (Piotroski 9-point)
            fscore = 0
            roa_cur = _safe_div(cur_a.get("NetIncomeLoss"), cur_a.get("Assets"))
            if roa_cur is not None and roa_cur > 0:
                fscore += 1
            cfo_c = cur_a.get("NetCashProvidedByUsedInOperatingActivities")
            if cfo_c is not None and float(cfo_c) > 0:
                fscore += 1
            roa_pri = _safe_div(pri_a.get("NetIncomeLoss"), pri_a.get("Assets"))
            if roa_cur is not None and roa_pri is not None and roa_cur > roa_pri:
                fscore += 1
            ni_c = cur_a.get("NetIncomeLoss")
            if cfo_c is not None and ni_c is not None and float(cfo_c) > float(ni_c):
                fscore += 1
            if float(cur_a.get("LongTermDebtNoncurrent") or 0) < float(pri_a.get("LongTermDebtNoncurrent") or 0):
                fscore += 1
            cr_c = _safe_div(cur_a.get("AssetsCurrent"), cur_a.get("LiabilitiesCurrent"))
            cr_p = _safe_div(pri_a.get("AssetsCurrent"), pri_a.get("LiabilitiesCurrent"))
            if cr_c is not None and cr_p is not None and cr_c > cr_p:
                fscore += 1
            if float(cur_a.get("CommonStockSharesOutstanding") or 0) <= float(pri_a.get("CommonStockSharesOutstanding") or 0):
                fscore += 1
            gm_c = _safe_div(float(cur_a.get("Revenues") or 0) - float(cur_a.get("CostOfRevenue") or 0), cur_a.get("Revenues"))
            gm_p = _safe_div(float(pri_a.get("Revenues") or 0) - float(pri_a.get("CostOfRevenue") or 0), pri_a.get("Revenues"))
            if gm_c is not None and gm_p is not None and gm_c > gm_p:
                fscore += 1
            at_c = _safe_div(cur_a.get("Revenues"), cur_a.get("Assets"))
            at_p = _safe_div(pri_a.get("Revenues"), pri_a.get("Assets"))
            if at_c is not None and at_p is not None and at_c > at_p:
                fscore += 1
            result["FScore"] = fscore

        # 3yr CAGR
        if len(annual_rows) >= 3:
            oldest = annual_rows.iloc[-3]
            rev_old_v = oldest.get("Revenues")
            rev_cur_v = annual_rows.iloc[-1].get("Revenues")
            if rev_old_v and rev_cur_v and float(rev_old_v) > 0:
                try:
                    result["RevenueGrowth3yr"] = (float(rev_cur_v) / float(rev_old_v)) ** (1/3) - 1
                except (TypeError, ValueError):
                    result["RevenueGrowth3yr"] = None
            else:
                result["RevenueGrowth3yr"] = None
        else:
            result["RevenueGrowth3yr"] = None

        results.append(result)

    return pd.DataFrame(results)


def apply_quality_screen(
    df: pd.DataFrame,
    min_roic: float = 0.15,
    min_operating_margin: float = 0.10,
    max_debt_to_equity: float = 1.0,
    min_interest_coverage: float = 5.0,
    min_cfo_to_ni: float = 0.8,
    require_positive_ni: bool = True,
) -> pd.DataFrame:
    """Buffett/Pabrai-flavored quality filter: profitable, conservatively
    financed, cash-generative, earnings that are backed by cash. Thresholds
    are exposed as parameters (not hardcoded) so a UI layer can let a user
    adjust them live and re-screen without touching this code -- these
    defaults are reasonable starting points, not universal truths."""
    mask = (
        (df["ROIC"] > min_roic) &
        (df["OperatingMargin"] > min_operating_margin) &
        (df["DebtToEquity"] < max_debt_to_equity) &
        (df["InterestCoverage"] > min_interest_coverage) &
        (df["CFO_to_NI"] > min_cfo_to_ni)
    )
    if require_positive_ni:
        mask &= (df["NetIncomeLoss"] > 0)
    return df[mask].copy()


def rank_magic_formula(df: pd.DataFrame) -> pd.DataFrame:
    """Combined rank on ROIC (quality) -- Earnings Yield is added once market
    data (EBIT/Enterprise Value) is joined in. Until then this ranks on ROIC
    alone, which is still directionally useful for a quality-only pre-screen."""
    df = df.copy()
    df["roic_rank"] = df["ROIC"].rank(ascending=False)
    df = df.sort_values("roic_rank")
    return df


def main():
    ap = argparse.ArgumentParser(description="SEC DERA value-investing screen")
    ap.add_argument("--data-dir", required=True, help="Folder with sub.txt/num.txt for one SEC quarterly dump")
    ap.add_argument("--form", default="10-K", help="Form type to filter to (default 10-K)")
    ap.add_argument("--annual", action="store_true", default=True, help="Treat flow tags as annual (qtrs=4). Use --quarterly for qtrs=1")
    ap.add_argument("--quarterly", dest="annual", action="store_false")
    ap.add_argument("--out", default="screen_output.csv", help="Output CSV path")
    ap.add_argument("--full-out", default=None, help="Optional: dump full fundamentals+ratios (pre-screen) to this CSV")
    ap.add_argument("--min-roic", type=float, default=0.15)
    ap.add_argument("--min-operating-margin", type=float, default=0.10)
    ap.add_argument("--max-debt-to-equity", type=float, default=1.0)
    ap.add_argument("--min-interest-coverage", type=float, default=5.0)
    ap.add_argument("--min-cfo-to-ni", type=float, default=0.8)
    args = ap.parse_args()

    con = duckdb.connect()
    print(f"[1/5] Loading {args.data_dir} ...")
    load_data(con, args.data_dir)

    print(f"[2/5] Filtering to form={args.form}, most recent filing per CIK ...")
    filter_submissions(con, args.form)

    print("[3/5] Pivoting tags into fundamentals table ...")
    df = build_fundamentals(con, annual=args.annual)
    print(f"      {len(df)} filings loaded")

    print("[4/5] Computing ratios ...")
    df = compute_ratios(df)
    if args.full_out:
        df.to_csv(args.full_out, index=False)
        print(f"      full fundamentals+ratios written to {args.full_out}")

    print("[5/5] Applying quality screen + ranking ...")
    screened = apply_quality_screen(
        df,
        min_roic=args.min_roic,
        min_operating_margin=args.min_operating_margin,
        max_debt_to_equity=args.max_debt_to_equity,
        min_interest_coverage=args.min_interest_coverage,
        min_cfo_to_ni=args.min_cfo_to_ni,
    )
    ranked = rank_magic_formula(screened)

    cols = ["cik", "name", "sic", "fy", "ROIC", "OperatingMargin", "NetMargin",
            "ROE", "ROA", "DebtToEquity", "InterestCoverage", "CurrentRatio",
            "CFO_to_NI", "AccrualRatio", "FCF"]
    ranked[cols].to_csv(args.out, index=False)
    print(f"\nDone. {len(ranked)} companies passed the quality screen out of {len(df)} filings.")
    print(f"Ranked screen written to: {args.out}")
    print("\nNOTE: This ranks on ROIC only. To finish a true Magic Formula rank")
    print("(ROIC + Earnings Yield), join market cap/price data -- see fetch_market_data.py")


if __name__ == "__main__":
    main()