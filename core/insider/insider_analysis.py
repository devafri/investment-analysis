"""Insider trading analysis module based on Cohen, Malloy, and Pomorski (2010)
"Decoding Inside Information."

Classifies SEC insider transactions as *routine* or *opportunistic* using
the paper's calendar-month persistence algorithm, then aggregates into
firm-level signals for return prediction.

Overview
--------
1. **load_and_process_data()** — reads SEC EDGAR quarterly ZIP files
   (NONDERIV_TRANS.txt, SUBMISSION.txt, REPORTINGOWNER.txt), filters to
   open-market purchases and sales, and merges into one clean DataFrame.
2. **classify_insiders()** — for each (insider, calendar month) pair, counts
   unique years with ≥1 trade.  If ≥3 years, all that insider's trades in that
   month are ROUTINE; otherwise OPPORTUNISTIC.
3. **create_signals()** — pivots to firm-month level: counts of opportunistic
   buys, opportunistic sells, routine buys, routine sells per (ISSUERCIK, month).

Reference
--------
Cohen, L., Malloy, C., & Pomorski, Ł. (2010). Decoding Inside Information.
NBER Working Paper No. 16454.  https://doi.org/10.3386/w16454
"""

import logging
import os
import re
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Canonical column names we use internally (normalised from whatever the
# SEC file actually provides).  Form 3/4/5 ("form345") and DERA use
# slightly different column names — we map both to these.
CANONICAL_COLS = [
    "ACCESSION_NUMBER", "TRANS_DATE", "TRANSACTION_CODE",
    "TRANS_SHARES", "TRANS_PRICEPERSHARE", "TRANS_ACQUIRED_DISP_CD",
    "ISSUERCIK", "ISSUERTRADINGSYMBOL",
    "RPTOWNERCIK", "RPTOWNERNAME", "RPTOWNERRELATIONSHIP", "RPTOWNERTITLE",
]

# Maps from canonical name to all known SEC column names (case-insensitive
# match — the first matching column in the actual file wins).
COLUMN_ALIASES = {
    "ACCESSION_NUMBER":     ["ACCESSION_NUMBER"],
    "TRANS_DATE":           ["TRANSACTION_DATE", "TRANS_DATE"],
    "TRANSACTION_CODE":     ["TRANSACTION_CODE"],
    "TRANS_SHARES":         ["TRANSACTION_SHARES", "TRANS_SHARES"],
    "TRANS_PRICEPERSHARE":  ["TRANSACTION_PRICE_PER_SHARE", "TRANS_PRICEPERSHARE"],
    "TRANS_ACQUIRED_DISP_CD": ["TRANSACTION_ACQUIRED_DISP_CD", "TRANS_ACQUIRED_DISP_CD"],
    "ISSUERCIK":            ["ISSUERCIK"],
    "ISSUERTRADINGSYMBOL":  ["ISSUERTRADINGSYMBOL"],
    "RPTOWNERCIK":          ["RPTOWNERCIK"],
    "RPTOWNERNAME":         ["RPTOWNERNAME"],
    "RPTOWNERRELATIONSHIP": ["RPTOWNERRELATIONSHIP"],
    "RPTOWNERTITLE":        ["RPTOWNERTITLE"],
}

# Only open-market purchases and sales — no option exercises, grants, gifts.
VALID_TRANSACTION_CODES = {"P", "S"}

# Minimum unique years trading in a given calendar month for an insider to
# be classified as a routine trader (Cohen et al. use 3).
MIN_ROUTINE_YEARS = 3

# ---------------------------------------------------------------------------
# In-memory cache for classification results (keyed by a hash of the
# (insider_cik, month, year) tuples so re-runs on unchanged data are instant).
# ---------------------------------------------------------------------------
_cache: Dict[int, pd.DataFrame] = {}


def _cache_key(df: pd.DataFrame) -> int:
    """Stable hash of the (RPTOWNERCIK, TRANS_DATE) pairs used for caching."""
    sample = (
        df[["RPTOWNERCIK", "TRANS_DATE"]]
        .drop_duplicates()
        .to_records(index=False)
    )
    return hash(sample.tobytes())


# ===================================================================
# 1. Data Ingestion
# ===================================================================


def _normalize_columns(df: pd.DataFrame, _file_label: str = "") -> pd.DataFrame:
    """Rename columns from SEC files to our canonical names.

    SEC Form 3/4/5 data uses names like ``TRANSACTION_DATE`` and
    ``TRANSACTION_SHARES`` while the DERA quarterly data uses ``TRANS_DATE``
    and ``TRANS_SHARES``.  This function maps both to the canonical names
    so downstream code doesn't care which format the file used.
    """
    if df.empty:
        return df
    # Build a case-insensitive lookup: lowercase actual → actual
    actual_map = {c.strip().upper(): c for c in df.columns}
    renames = {}
    for canon, aliases in COLUMN_ALIASES.items():
        for alias in aliases:
            key = alias.upper()
            if key in actual_map and actual_map[key] != canon:
                renames[actual_map[key]] = canon
                break
    if renames:
        df = df.rename(columns=renames)
    return df


def _read_sec_file(zip_archive: zipfile.ZipFile, filename: str,
                   required: bool = True) -> pd.DataFrame:
    """Read one tab-delimited file from a SEC quarterly ZIP.  Reads ALL
    columns (different SEC datasets use different column names — we
    normalise afterwards)."""
    try:
        with zip_archive.open(filename) as f:
            return pd.read_csv(
                f, sep="\t", dtype=str, encoding="latin-1", low_memory=False,
            )
    except KeyError:
        if required:
            raise FileNotFoundError(
                f"{filename} not found in {zip_archive.filename}"
            )
        logger.warning("%s not found in %s — skipping",
                       filename, os.path.basename(zip_archive.filename))
        return pd.DataFrame()


def _parse_quarter_from_filename(path: str) -> str:
    """Extract a quarter label from a ZIP filename.  Tries common patterns
    (2024q1, 2024_q1, 2024-Q1) and falls back to the bare filename."""
    name = os.path.basename(path).lower()
    for pat in [r"(\d{4})\s*q\s*([1-4])", r"(\d{4})[-_]q\s*([1-4])"]:
        m = re.search(pat, name)
        if m:
            return f"{m.group(1)}q{m.group(2)}"
    m = re.search(r"(20\d{2})", name)
    if m:
        return m.group(1)
    return name.replace(".zip", "")


def _safe_read_tsv(filepath: Path) -> pd.DataFrame:
    """Read a TSV file from disk, returning empty DataFrame if not found."""
    if not filepath.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(
            filepath, sep="\t", dtype=str, encoding="latin-1", low_memory=False,
        )
    except Exception:
        return pd.DataFrame()


def _find_data_sources(data_dir: str) -> List[Tuple[Path, str]]:
    """Discover SEC insider-transaction data sources in *data_dir*.

    Returns ``(path, label)`` tuples — *path* is either a directory or ZIP,
    *label* is a human-readable label.  Search order:
    1. data_dir itself (if it directly contains NONDERIV_TRANS.txt)
    2. Subdirectories containing the TXT files
    3. ZIP files with year-quarter naming (e.g. 2024q1.zip)
    4. ALL remaining ZIP files (fallback — arbitrary names)
    """
    data_path = Path(data_dir).expanduser()
    if not data_path.exists():
        raise FileNotFoundError(f"Directory not found: {data_dir}")

    sources: List[Tuple[Path, str]] = []
    seen: set = set()

    # 1 — data_dir itself (unzipped)
    if (data_path / "NONDERIV_TRANS.txt").exists():
        label = _parse_quarter_from_filename(data_path.name)
        sources.append((data_path, label))
        seen.add(data_path)

    # 2 — subdirectories with the TXT files
    for sub in sorted(data_path.iterdir()):
        if sub.is_dir() and (sub / "NONDERIV_TRANS.txt").exists():
            label = _parse_quarter_from_filename(sub.name)
            sources.append((sub, label))
            seen.add(sub)

    # 3 — ZIP files with year-quarter pattern (e.g. 2024q1.zip)
    for p in sorted(data_path.glob("*.zip")):
        if p in seen:
            continue
        if re.search(r"(20\d{2})\s*q", p.name, re.IGNORECASE):
            label = _parse_quarter_from_filename(p.name)
            sources.append((p, label))
            seen.add(p)

    # 4 — ALL remaining ZIP files (catch-all for any naming convention)
    for p in sorted(data_path.glob("*.zip")):
        if p in seen:
            continue
        label = _parse_quarter_from_filename(p.name)
        sources.append((p, label))
        seen.add(p)

    return sources


def load_and_process_data(
    data_dir: str,
    start_year: int = 2016,
    end_year: int = 2026,
) -> pd.DataFrame:
    """Ingest SEC EDGAR insider-transaction data from *data_dir*.

    Accepts any of these layouts:
    - A directory containing the three TXT files directly
    - Subdirectories each containing the TXT files
    - ZIP archives (ANY naming convention) containing the TXT files

    The *start_year* / *end_year* filter is applied to the actual trade
    dates inside the files (NOT the filename), so it works regardless of
    ZIP naming.

    Returns a merged DataFrame with columns:
        ACCESSION_NUMBER, TRANS_DATE, TRANSACTION_CODE, TRANS_SHARES,
        TRANS_PRICEPERSHARE, ISSUERCIK, ISSUERTRADINGSYMBOL, RPTOWNERCIK,
        RPTOWNERNAME, RPTOWNERRELATIONSHIP, RPTOWNERTITLE, QUARTER
    """
    sources = _find_data_sources(data_dir)
    if not sources:
        raise FileNotFoundError(
            f"No insider-transaction data found in {data_dir}. "
            f"Place SEC EDGAR quarterly ZIP files (or unzipped folders "
            f"containing NONDERIV_TRANS.txt, SUBMISSION.txt, "
            f"REPORTINGOWNER.txt) in this directory."
        )

    all_frames: List[pd.DataFrame] = []
    total_rows = 0
    processed = 0

    for source_path, label in sources:
        logger.info("Processing %s …", label)
        try:
            if source_path.is_dir():
                trans = pd.read_csv(
                    source_path / "NONDERIV_TRANS.txt",
                    sep="\t", dtype=str, encoding="latin-1", low_memory=False,
                )
                sub = _safe_read_tsv(source_path / "SUBMISSION.txt")
                owners = _safe_read_tsv(source_path / "REPORTINGOWNER.txt")
            else:
                with zipfile.ZipFile(source_path) as zf:
                    trans = _read_sec_file(zf, "NONDERIV_TRANS.txt")
                    sub = _read_sec_file(zf, "SUBMISSION.txt")
                    owners = _read_sec_file(zf, "REPORTINGOWNER.txt")

            # Normalise column names (Form 3/4/5 uses TRANSACTION_DATE etc.,
            # DERA uses TRANS_DATE — we want the canonical names).
            trans = _normalize_columns(trans, label)
            sub = _normalize_columns(sub, label)
            owners = _normalize_columns(owners, label)

            if trans.empty:
                logger.warning("%s: no transactions — skipping", label)
                continue

            n_raw = len(trans)
            if n_raw == 0:
                continue

            # --- Filter to open-market purchases & sales only ---
            trans_codes = trans["TRANSACTION_CODE"].str.strip().str.upper()
            trans = trans[trans_codes.isin(VALID_TRANSACTION_CODES)].copy()
            n_after_code = len(trans)
            if n_after_code == 0:
                logger.info(
                    "  %s: %,d rows, 0 after code filter "
                    "(unique codes: %s)",
                    label, n_raw,
                    ", ".join(sorted(set(trans_codes.dropna()))),
                )
                continue

            # --- Parse dates (try multiple formats) ---
            date_strs = trans["TRANS_DATE"].str.strip()
            trans["TRANS_DATE"] = pd.to_datetime(
                date_strs, format="%d-%b-%Y", errors="coerce",
            )
            # If most dates failed, try 2-digit year variant
            na_date_pct = trans["TRANS_DATE"].isna().mean()
            if na_date_pct > 0.5:
                trans["TRANS_DATE"] = trans["TRANS_DATE"].fillna(
                    pd.to_datetime(date_strs, format="%d-%b-%y", errors="coerce")
                )
            trans["TRANS_SHARES"] = pd.to_numeric(
                trans["TRANS_SHARES"], errors="coerce",
            )
            trans["TRANS_PRICEPERSHARE"] = pd.to_numeric(
                trans["TRANS_PRICEPERSHARE"], errors="coerce",
            )
            n_before_drop = len(trans)
            trans = trans.dropna(subset=["TRANS_DATE", "TRANS_SHARES"])
            n_after_drop = len(trans)
            if n_after_drop == 0 and n_before_drop > 0:
                logger.info(
                    "  %s: %,d rows after code filter, 0 after date/share parse",
                    label, n_before_drop,
                )
                continue

            # --- Year filter (applied to trade dates, NOT filenames) ---
            trans_year = trans["TRANS_DATE"].dt.year
            trans = trans[
                (trans_year >= start_year) & (trans_year <= end_year)
            ].copy()
            if trans.empty:
                continue

            # --- Merge with submission & owner metadata ---
            if not sub.empty and not owners.empty:
                merged = trans.merge(
                    sub[["ACCESSION_NUMBER", "ISSUERCIK", "ISSUERTRADINGSYMBOL"]],
                    on="ACCESSION_NUMBER", how="left",
                )
                merged = merged.merge(
                    owners[["ACCESSION_NUMBER", "RPTOWNERCIK", "RPTOWNERNAME",
                            "RPTOWNERRELATIONSHIP", "RPTOWNERTITLE"]],
                    on="ACCESSION_NUMBER", how="left",
                )
            else:
                merged = trans
                for col in ["ISSUERCIK", "RPTOWNERCIK"]:
                    if col not in merged.columns:
                        merged[col] = None

            merged["QUARTER"] = label
            all_frames.append(merged)
            total_rows += len(merged)
            processed += 1
            logger.info("  %s: %,d open-market trades (after year filter)", label, len(merged))

        except Exception:
            logger.exception("Failed to process %s — skipping", label)

    if not all_frames:
        raise ValueError(
            f"No valid transaction data found in {data_dir}. "
            f"Checked {len(sources)} source(s). Ensure the files contain "
            f"open-market purchases (P) or sales (S) with valid dates "
            f"between {start_year} and {end_year}."
        )

    result = pd.concat(all_frames, ignore_index=True)
    logger.info(
        "Loaded %,d trades across %d sources (%d unique insiders, %d firms)",
        total_rows, processed,
        result["RPTOWNERCIK"].nunique() if "RPTOWNERCIK" in result.columns else 0,
        result["ISSUERCIK"].nunique() if "ISSUERCIK" in result.columns else 0,
    )
    return result


# ===================================================================
# 2. Insider Classification (routine vs opportunistic)
# ===================================================================


def classify_insiders(
    df: pd.DataFrame,
    min_years: int = MIN_ROUTINE_YEARS,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Classify each transaction as ROUTINE or OPPORTUNISTIC.

    Algorithm (Cohen, Malloy & Pomorski 2010)
    -----------------------------------------
    1. Group trades by (RPTOWNERCIK, calendar month).
    2. Count the number of **unique years** in which that insider traded
       during that calendar month.
    3. If unique years ≥ *min_years*, ALL trades by that insider in that
       calendar month are ROUTINE.  Otherwise OPPORTUNISTIC.

    Parameters
    ----------
    df : DataFrame
        Must have columns RPTOWNERCIK and TRANS_DATE (datetime).
    min_years : int
        Threshold for routine classification (default 3).
    use_cache : bool
        If True, cache the result and return a copy on subsequent calls
        with the same data.

    Returns
    -------
    DataFrame
        Copy of *df* with an added TRADE_TYPE column ('ROUTINE' or
        'OPPORTUNISTIC') and ROUTINE_YEARS (int, ≥1).
    """
    if df.empty:
        df = df.copy()
        df["TRADE_TYPE"] = "OPPORTUNISTIC"
        df["ROUTINE_YEARS"] = 0
        return df

    # --- Cache check ---
    ck = _cache_key(df)
    if use_cache and ck in _cache:
        return _cache[ck].copy()

    # --- Build (insider, month) → set of years ---
    df_work = df[["RPTOWNERCIK", "TRANS_DATE"]].dropna().copy()
    df_work["_month"] = df_work["TRANS_DATE"].dt.month
    df_work["_year"] = df_work["TRANS_DATE"].dt.year

    # Group by (insider, month) and collect the set of unique years
    grouped = (
        df_work.groupby(["RPTOWNERCIK", "_month"])["_year"]
        .apply(lambda x: set(x))
    )

    # Map (insider, month) → (routine_years, is_routine)
    routine_map: Dict[Tuple[str, int], Tuple[int, bool]] = {}
    for (cik, month), years in grouped.items():
        ny = len(years)
        routine_map[(str(cik), int(month))] = (ny, ny >= min_years)

    # --- Apply classification to every row (vectorised via .map) ---
    result = df.copy()
    result["_month"] = result["TRANS_DATE"].dt.month
    result["_key"] = list(zip(
        result["RPTOWNERCIK"].astype(str),
        result["_month"].astype(int),
    ))

    mapped = result["_key"].map(routine_map)
    result["ROUTINE_YEARS"] = mapped.apply(lambda v: v[0] if v is not None else 0)
    result["TRADE_TYPE"] = result["ROUTINE_YEARS"].apply(
        lambda y: "ROUTINE" if y >= min_years else "OPPORTUNISTIC"
    )

    result.drop(columns=["_month", "_key"], inplace=True)

    # --- Cache and return ---
    if use_cache:
        _cache[ck] = result.copy()
    return result


# ===================================================================
# 3. Firm-Level Signal Aggregation
# ===================================================================


def create_signals(
    df: pd.DataFrame,
    aggregate_by: str = "count",  # "count" | "shares" | "value"
) -> pd.DataFrame:
    """Aggregate classified insider trades to firm-month signals.

    Parameters
    ----------
    df : DataFrame
        Must have TRADE_TYPE, ISSUERCIK, TRANS_DATE, TRANSACTION_CODE,
        and TRANS_SHARES (and TRANS_PRICEPERSHARE if aggregate_by='value').
    aggregate_by : str
        'count'  — number of trades
        'shares' — total shares traded
        'value'  — total dollar value (shares × price)

    Returns
    -------
    DataFrame
        Indexed by (ISSUERCIK, YEAR, MONTH) with columns:
        OPP_BUY, OPP_SELL, ROUTINE_BUY, ROUTINE_SELL,
        OPP_NET (OPP_BUY − OPP_SELL), ROUTINE_NET.
    """
    df = df.copy()
    df["YEAR"] = df["TRANS_DATE"].dt.year
    df["MONTH"] = df["TRANS_DATE"].dt.month

    if aggregate_by == "value":
        df["_metric"] = df["TRANS_SHARES"].fillna(0) * df["TRANS_PRICEPERSHARE"].fillna(0)
    elif aggregate_by == "shares":
        df["_metric"] = df["TRANS_SHARES"].fillna(0)
    else:
        df["_metric"] = 1  # count

    # Build a composite key: TRADE_TYPE + transaction direction
    df["_signal_key"] = (
        df["TRADE_TYPE"].str[:4]  # "ROUT" or "OPPO"
        + "_"
        + df["TRANSACTION_CODE"].str.strip().str.upper().map({"P": "BUY", "S": "SELL"})
    )

    signals = (
        df.groupby(["ISSUERCIK", "YEAR", "MONTH", "_signal_key"])["_metric"]
        .sum()
        .unstack(fill_value=0)
    )

    # Ensure all four expected columns exist
    for col in ["OPPO_BUY", "OPPO_SELL", "ROUT_BUY", "ROUT_SELL"]:
        if col not in signals.columns:
            signals[col] = 0

    signals["OPP_NET"] = signals["OPPO_BUY"] - signals["OPPO_SELL"]
    signals["ROUTINE_NET"] = signals["ROUT_BUY"] - signals["ROUT_SELL"]

    return signals.reset_index()


# ===================================================================
# 4. Summary Statistics & Analysis
# ===================================================================


def get_summary_stats(df: pd.DataFrame) -> Dict[str, Any]:
    """Return summary statistics for classified insider trades."""
    if df.empty or "TRADE_TYPE" not in df.columns:
        return {"error": "No classified trade data available"}

    total = len(df)
    opp = (df["TRADE_TYPE"] == "OPPORTUNISTIC").sum()
    routine = total - opp

    buys = (df["TRANSACTION_CODE"].str.strip().str.upper() == "P").sum()
    sells = total - buys

    opp_buys = ((df["TRADE_TYPE"] == "OPPORTUNISTIC") &
                (df["TRANSACTION_CODE"].str.strip().str.upper() == "P")).sum()
    opp_sells = opp - opp_buys

    return {
        "total_trades": total,
        "opportunistic_trades": int(opp),
        "routine_trades": int(routine),
        "pct_opportunistic": round(opp / total * 100, 1) if total else 0,
        "pct_routine": round(routine / total * 100, 1) if total else 0,
        "total_buys": int(buys),
        "total_sells": int(sells),
        "opp_buys": int(opp_buys),
        "opp_sells": int(opp_sells),
        "unique_insiders": int(df["RPTOWNERCIK"].nunique()) if "RPTOWNERCIK" in df.columns else 0,
        "unique_routine_insiders": int(
            df[df["TRADE_TYPE"] == "ROUTINE"]["RPTOWNERCIK"].nunique()
        ) if "RPTOWNERCIK" in df.columns else 0,
        "unique_opportunistic_insiders": int(
            df[df["TRADE_TYPE"] == "OPPORTUNISTIC"]["RPTOWNERCIK"].nunique()
        ) if "RPTOWNERCIK" in df.columns else 0,
        "unique_firms": int(df["ISSUERCIK"].nunique()) if "ISSUERCIK" in df.columns else 0,
    }


# ===================================================================
# 5. InsiderTradingAnalyzer — convenience class
# ===================================================================


class InsiderTradingAnalyzer:
    """Convenience wrapper that ties ingestion, classification, and signal
    creation together with optional caching."""

    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self._transactions: Optional[pd.DataFrame] = None
        self._classified: Optional[pd.DataFrame] = None
        self._signals: Optional[pd.DataFrame] = None

    def load_all_quarters(
        self, start_year: int = 2016, end_year: int = 2026,
    ) -> pd.DataFrame:
        """Ingest all quarterly ZIP files between *start_year* and *end_year*."""
        self._transactions = load_and_process_data(
            self.data_dir, start_year=start_year, end_year=end_year,
        )
        self._classified = None
        self._signals = None
        return self._transactions

    def classify_insiders(self, df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Classify each trade as ROUTINE or OPPORTUNISTIC."""
        if df is not None:
            self._transactions = df
        if self._transactions is None:
            raise ValueError("No transaction data loaded. Call load_all_quarters() first.")
        self._classified = classify_insiders(self._transactions)
        self._signals = None
        return self._classified

    def create_signals(
        self, df: Optional[pd.DataFrame] = None, aggregate_by: str = "count",
    ) -> pd.DataFrame:
        """Aggregate to firm-month signals."""
        if df is not None:
            self._classified = df
        if self._classified is None:
            raise ValueError("No classified data. Call classify_insiders() first.")
        self._signals = create_signals(self._classified, aggregate_by=aggregate_by)
        return self._signals

    def run_analysis(self, signals: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """Return summary statistics and firm-level signals."""
        if signals is not None:
            self._signals = signals
        if self._classified is None:
            raise ValueError("No classified data. Call classify_insiders() first.")
        return {
            "summary_stats": get_summary_stats(self._classified),
            "signals": self._signals,
        }


# ===================================================================
# 6. Regression helpers (optional — requires statsmodels + return data)
# ===================================================================


def run_fama_macbeth(
    signals: pd.DataFrame,
    returns: pd.DataFrame,
    controls: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Run Fama-MacBeth cross-sectional regressions of month-t+1 returns on
    insider trading signals.

    Parameters
    ----------
    signals : DataFrame
        From create_signals().  Must have ISSUERCIK, YEAR, MONTH.
    returns : DataFrame
        Must have ISSUERCIK (or PERMCO), YEAR, MONTH, RET (monthly return).
    controls : list[str], optional
        Additional control variable column names in *returns*.

    Returns
    -------
    dict with coefficients, t-stats, R², N (number of months).
    """
    try:
        import statsmodels.api as sm
    except ImportError:
        return {"error": "statsmodels is required for regression analysis. "
                         "Install with: pip install statsmodels"}

    # Merge signals with next-month returns
    merged = signals.merge(returns, on=["ISSUERCIK", "YEAR", "MONTH"], how="inner")
    if merged.empty:
        return {"error": "No overlapping data between signals and returns"}

    # Normalize signals by market cap or just use as-is
    y = merged.pop("RET") if "RET" in merged.columns else None
    if y is None:
        return {"error": "No RET column in returns DataFrame"}

    X_cols = ["OPP_NET", "ROUTINE_NET"]
    if controls:
        X_cols.extend([c for c in controls if c in merged.columns])

    X = merged[X_cols].copy()
    X = sm.add_constant(X)

    # Fama-MacBeth: cross-sectional regression each month, then average
    monthly_results = []
    for (year, month), group in merged.groupby(["YEAR", "MONTH"]):
        idx = group.index
        y_m = y.loc[idx]
        X_m = X.loc[idx].dropna()
        common = y_m.index.intersection(X_m.index)
        if len(common) < 20:
            continue
        try:
            model = sm.OLS(y_m.loc[common], X_m.loc[common]).fit()
            monthly_results.append({
                "year": year, "month": month,
                "n": len(common),
                **model.params.to_dict(),
                "rsq": model.rsquared,
            })
        except Exception:
            continue

    if not monthly_results:
        return {"error": "No monthly regressions could be estimated"}

    fm_df = pd.DataFrame(monthly_results)
    param_cols = [c for c in fm_df.columns if c not in ("year", "month", "n", "rsq")]
    coefs = fm_df[param_cols].mean().to_dict()
    tstats = (fm_df[param_cols].mean() /
              (fm_df[param_cols].std() / np.sqrt(len(fm_df)))).to_dict()

    return {
        "coefficients": coefs,
        "t_statistics": tstats,
        "r_squared_avg": fm_df["rsq"].mean(),
        "n_months": len(fm_df),
        "avg_firms_per_month": fm_df["n"].mean(),
    }


# ===================================================================
# 7. DuckDB persistence (integrate with the main screening app)
# ===================================================================


def persist_insider_trades(con, df: pd.DataFrame) -> int:
    """Store classified insider trades in the persistent DuckDB table
    ``insider_trades``.  Deduplicates by ACCESSION_NUMBER so re-ingesting
    the same quarter is idempotent.

    Returns the number of new rows inserted.
    """
    if df.empty or "TRADE_TYPE" not in df.columns:
        return 0

    cols = [
        "ACCESSION_NUMBER", "TRANS_DATE", "TRANSACTION_CODE",
        "TRANS_SHARES", "TRANS_PRICEPERSHARE", "ISSUERCIK",
        "ISSUERTRADINGSYMBOL", "RPTOWNERCIK", "RPTOWNERNAME",
        "RPTOWNERRELATIONSHIP", "RPTOWNERTITLE",
        "TRADE_TYPE", "ROUTINE_YEARS", "QUARTER",
    ]
    available = [c for c in cols if c in df.columns]
    batch = df[available].copy()

    con.register("_insider_batch", batch)
    con.execute("""
        CREATE TABLE IF NOT EXISTS insider_trades (
            accession_number VARCHAR PRIMARY KEY,
            trans_date TIMESTAMP,
            transaction_code VARCHAR,
            trans_shares DOUBLE,
            trans_price_per_share DOUBLE,
            issuer_cik VARCHAR,
            issuer_trading_symbol VARCHAR,
            rpt_owner_cik VARCHAR,
            rpt_owner_name VARCHAR,
            rpt_owner_relationship VARCHAR,
            rpt_owner_title VARCHAR,
            trade_type VARCHAR,
            routine_years INTEGER,
            quarter VARCHAR
        )
    """)
    before = con.execute("SELECT COUNT(*) FROM insider_trades").fetchone()[0]
    con.execute("""
        INSERT OR IGNORE INTO insider_trades
        SELECT * FROM _insider_batch
    """)
    con.unregister("_insider_batch")
    after = con.execute("SELECT COUNT(*) FROM insider_trades").fetchone()[0]
    return after - before


def resolve_company_name(cik: str, con) -> str:
    """Look up a company name from the existing DuckDB fundamentals data.
    Falls back to the insider trade's ticker symbol if no name is found."""
    try:
        row = con.execute(
            "SELECT name FROM fundamentals_history "
            "WHERE CAST(cik AS VARCHAR) = ? LIMIT 1",
            [str(cik)],
        ).fetchone()
        if row and row[0]:
            return str(row[0])
    except Exception:
        pass
    # Fallback: try the sub table from the most recent ingest
    try:
        row = con.execute(
            "SELECT name FROM sub WHERE CAST(cik AS VARCHAR) = ? LIMIT 1",
            [str(cik)],
        ).fetchone()
        if row and row[0]:
            return str(row[0])
    except Exception:
        pass
    return None


def get_insider_trades_for_cik(
    cik: str, con, limit: int = 50,
) -> pd.DataFrame:
    """Return recent insider trades for a company, with company name resolved.
    Most recent trades first."""
    df = con.execute(
        "SELECT * FROM insider_trades "
        "WHERE CAST(issuer_cik AS VARCHAR) = ? "
        "ORDER BY trans_date DESC LIMIT ?",
        [str(cik), limit],
    ).fetchdf()
    if df.empty:
        return df

    # Resolve company name
    name = resolve_company_name(cik, con)
    if name:
        df["COMPANY_NAME"] = name

    return df


def get_insider_summary_for_cik(cik: str, con) -> dict:
    """Return summary statistics of insider activity for a company."""
    row = con.execute(
        "SELECT "
        "  COUNT(*) AS total_trades,"
        "  SUM(CASE WHEN trade_type = 'ROUTINE' THEN 1 ELSE 0 END) AS routine_trades,"
        "  SUM(CASE WHEN trade_type = 'OPPORTUNISTIC' THEN 1 ELSE 0 END) AS opp_trades,"
        "  SUM(CASE WHEN trade_type = 'OPPORTUNISTIC' "
        "      AND transaction_code = 'P' THEN 1 ELSE 0 END) AS opp_buys,"
        "  SUM(CASE WHEN trade_type = 'OPPORTUNISTIC' "
        "      AND transaction_code = 'S' THEN 1 ELSE 0 END) AS opp_sells,"
        "  MIN(trans_date) AS first_trade,"
        "  MAX(trans_date) AS last_trade "
        "FROM insider_trades "
        "WHERE CAST(issuer_cik AS VARCHAR) = ?",
        [str(cik)],
    ).fetchone()
    if not row or row[0] == 0:
        return {"total_trades": 0}
    return {
        "total_trades": int(row[0]),
        "routine_trades": int(row[1]),
        "opp_trades": int(row[2]),
        "opp_buys": int(row[3]),
        "opp_sells": int(row[4]),
        "first_trade": str(row[5]) if row[5] else None,
        "last_trade": str(row[6]) if row[6] else None,
    }


def has_insider_data(con) -> bool:
    """Check whether insider trade data has been ingested."""
    try:
        return con.execute(
            "SELECT COUNT(*) FROM information_schema.tables "
            "WHERE table_name = 'insider_trades'"
        ).fetchone()[0] > 0
    except Exception:
        return False


def get_aggregate_summary(con) -> dict:
    """Return aggregate summary statistics across ALL companies."""
    row = con.execute("""
        SELECT
            COUNT(*) AS total,
            SUM(CASE WHEN trade_type = 'ROUTINE' THEN 1 ELSE 0 END) AS routine,
            SUM(CASE WHEN trade_type = 'OPPORTUNISTIC' THEN 1 ELSE 0 END) AS opp,
            SUM(CASE WHEN trade_type = 'OPPORTUNISTIC' AND transaction_code = 'P'
                THEN 1 ELSE 0 END) AS opp_buys,
            SUM(CASE WHEN trade_type = 'OPPORTUNISTIC' AND transaction_code = 'S'
                THEN 1 ELSE 0 END) AS opp_sells,
            COUNT(DISTINCT rpt_owner_cik) AS insiders,
            COUNT(DISTINCT issuer_cik) AS firms,
            MIN(trans_date) AS first_date,
            MAX(trans_date) AS last_date
        FROM insider_trades
    """).fetchone()
    if not row or row[0] == 0:
        return {"total": 0}
    return {
        "total": int(row[0]), "routine": int(row[1]), "opp": int(row[2]),
        "opp_buys": int(row[3]), "opp_sells": int(row[4]),
        "insiders": int(row[5]), "firms": int(row[6]),
        "first_date": str(row[7])[:10] if row[7] else None,
        "last_date": str(row[8])[:10] if row[8] else None,
    }


def search_insider_trades(
    con,
    search: str = "",
    trade_type: str = "",  # "OPPORTUNISTIC", "ROUTINE", or "" for all
    code: str = "",        # "P" (buy), "S" (sell), or "" for all
    limit: int = 100,
    offset: int = 0,
) -> pd.DataFrame:
    """Search insider trades across all companies with optional filters.
    Returns trades joined with company names from fundamentals_history."""
    where = []
    params = []

    if search:
        where.append(
            "(LOWER(it.issuer_trading_symbol) LIKE ? "
            "OR LOWER(it.rpt_owner_name) LIKE ? "
            "OR CAST(it.issuer_cik AS VARCHAR) = ?)"
        )
        q = f"%{search.lower()}%"
        params.extend([q, q, search.strip()])

    if trade_type:
        where.append("it.trade_type = ?")
        params.append(trade_type)

    if code:
        where.append("it.transaction_code = ?")
        params.append(code)

    where_clause = ("WHERE " + " AND ".join(where)) if where else ""
    query = f"""
        SELECT it.*,
               COALESCE(fh.name, it.issuer_trading_symbol,
                        'CIK ' || it.issuer_cik) AS company_name
        FROM insider_trades it
        LEFT JOIN (
            SELECT DISTINCT cik, name
            FROM fundamentals_history
        ) fh ON CAST(fh.cik AS VARCHAR) = it.issuer_cik
        {where_clause}
        ORDER BY it.trans_date DESC
        LIMIT ? OFFSET ?
    """
    params.extend([limit, offset])
    return con.execute(query, params).fetchdf()


def get_top_companies(con, by: str = "opp_trades", limit: int = 10) -> pd.DataFrame:
    """Return companies ranked by insider activity."""
    metric = (
        "SUM(CASE WHEN trade_type = 'OPPORTUNISTIC' THEN 1 ELSE 0 END)"
        if by == "opp_trades" else
        "SUM(CASE WHEN trade_type = 'OPPORTUNISTIC' AND transaction_code = 'P' "
        "THEN 1 ELSE 0 END) - "
        "SUM(CASE WHEN trade_type = 'OPPORTUNISTIC' AND transaction_code = 'S' "
        "THEN 1 ELSE 0 END)"
    )
    return con.execute(f"""
        SELECT
            it.issuer_cik,
            COALESCE(fh.name, MAX(it.issuer_trading_symbol),
                     'CIK ' || it.issuer_cik) AS company_name,
            MAX(it.issuer_trading_symbol) AS ticker,
            COUNT(*) AS total_trades,
            SUM(CASE WHEN it.trade_type = 'OPPORTUNISTIC' THEN 1 ELSE 0 END) AS opp_trades,
            SUM(CASE WHEN it.trade_type = 'OPPORTUNISTIC' AND it.transaction_code = 'P'
                THEN 1 ELSE 0 END) AS opp_buys,
            SUM(CASE WHEN it.trade_type = 'OPPORTUNISTIC' AND it.transaction_code = 'S'
                THEN 1 ELSE 0 END) AS opp_sells,
            ({metric}) AS sort_metric
        FROM insider_trades it
        LEFT JOIN (
            SELECT DISTINCT cik, name FROM fundamentals_history
        ) fh ON CAST(fh.cik AS VARCHAR) = it.issuer_cik
        GROUP BY it.issuer_cik, fh.name
        ORDER BY sort_metric DESC
        LIMIT ?
    """, [limit]).fetchdf()
