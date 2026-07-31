"""Core screening logic: thresholds, loading TTM ratios from the accumulated
history, applying the quality screen, and pagination.
"""

import math
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlencode

import pandas as pd

from core.data_ingestion import get_db_connection
from core.exchange_filter import filter_to_major_exchanges
from core.types import Pagination, RowContext, ScreenInfo, ThresholdConfig
from core.valuation import apply_valuation, compute_margin_of_safety
from providers.market_data import join_market_data, attach_tickers
from providers.pipeline_imports import sec_screen, require_pipeline

PAGE_SIZE = 20

DEFAULT_THRESHOLDS = {
    "min_roic": 0.15,
    "min_operating_margin": 0.10,
    "max_debt_to_equity": 1.0,
    "min_interest_coverage": 5.0,
    "min_cfo_to_ni": 0.8,
    "min_revenue_growth": 0.0,
    "min_fscore": 0,
    "require_positive_ni": True,
    "major_exchanges_only": True,
    "exclude_financials": False,
    # DCF assumptions
    "growth_rate": 0.05,
    "discount_rate": 0.10,
    "terminal_growth_rate": 0.025,
    "projection_years": 10,
}

BOOLEAN_THRESHOLD_KEYS = {"require_positive_ni", "major_exchanges_only", "exclude_financials"}
PERCENT_INPUT_KEYS = {"min_roic", "min_operating_margin", "growth_rate", "discount_rate", "terminal_growth_rate", "min_revenue_growth"}  # NEW


def parse_thresholds(params: Dict[str, Any]) -> ThresholdConfig:
    thresholds = DEFAULT_THRESHOLDS.copy()
    for key, default in DEFAULT_THRESHOLDS.items():
        if key in BOOLEAN_THRESHOLD_KEYS:
            if key not in params:
                thresholds[key] = default
            else:
                thresholds[key] = params.get(key) in {"on", "true", "1", True, 1}
        else:
            raw_value = params.get(key)
            if raw_value is None:
                thresholds[key] = default
            else:
                thresholds[key] = float(raw_value)
    for key in PERCENT_INPUT_KEYS:  # CHANGED: was two separate `if` blocks for min_roic/min_operating_margin only
        if key in params and params[key] is not None:
            thresholds[key] = float(params[key]) / 100.0
    if "projection_years" in params and params["projection_years"] is not None:  # NEW
        thresholds["projection_years"] = int(float(params["projection_years"]))
    return thresholds


def build_query_string(params: Dict[str, Any], overrides: Optional[Dict[str, Any]] = None) -> str:
    merged = dict(params)
    if overrides:
        merged.update(overrides)
    return urlencode({k: v for k, v in merged.items() if v not in {None, ""}}, doseq=True)


def base_query_string(params: Dict[str, Any]) -> str:
    stripped = {k: v for k, v in params.items() if k not in {"sort", "order", "page"}}
    return urlencode(stripped, doseq=True)


# Module-level cache: the enriched TTM DataFrame is expensive to compute
# (groupby over 45K+ rows across 6K+ companies) but only changes when new
# data is ingested.  Cache it by fundamentals_history row count so every
# /screen and /company/{cik} request returns in milliseconds instead of
# seconds.
import threading as _threading
_enriched_cache: Optional[pd.DataFrame] = None
_cache_fingerprint: str = ""
_cache_lock = _threading.Lock()


def _invalidate_cache() -> None:
    global _enriched_cache, _cache_fingerprint
    with _cache_lock:
        _enriched_cache = None
        _cache_fingerprint = ""


def _get_db_fingerprint(con) -> str:
    """Return a stable fingerprint of the fundamentals_history table.

    Uses COUNT(*) + MAX(filed) + MAX(period) so the cache invalidates on
    same-row-count updates (e.g. a re-ingest that replaces overlapping
    quarters) as well as on row additions."""
    row = con.execute(
        "SELECT COUNT(*), COALESCE(MAX(filed), ''), COALESCE(MAX(period), '') "
        "FROM fundamentals_history"
    ).fetchone()
    return f"{row[0]}|{row[1]}|{row[2]}"


def load_cached_ratios() -> pd.DataFrame:
    """Load fundamentals_history, compute TTM, ratios, growth, and F-Score.
    Returns a fully enriched DataFrame ready for screening.  Cached until
    fundamentals_history changes."""
    global _enriched_cache, _cache_fingerprint

    require_pipeline()
    con = get_db_connection()
    try:
        if con.execute(
            "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = 'fundamentals_history'"
        ).fetchone()[0] == 0:
            raise FileNotFoundError("No cached screening data is available yet. Please ingest a data directory first.")

        if con.execute("SELECT COUNT(*) FROM fundamentals_history").fetchone()[0] == 0:
            raise FileNotFoundError("No cached screening data is available yet. Please ingest a data directory first.")

        fingerprint = _get_db_fingerprint(con)
        with _cache_lock:
            if _enriched_cache is not None and _cache_fingerprint == fingerprint:
                return _enriched_cache.copy()

        history = sec_screen.load_fundamentals_history(con)
        if history.empty:
            raise FileNotFoundError("No cached screening data is available yet. Please ingest a data directory first.")
        # Combined pass: TTM + growth + F-Score in a single groupby sweep.
        # This is the dominant cold-start cost.
        ttm = sec_screen.compute_ttm_with_enrichment(history)
        if ttm.empty:
            raise FileNotFoundError(
                "Filings were ingested, but none had a 10-K to anchor a TTM calculation. "
                "Make sure at least one annual 10-K is included in what you've ingested."
            )
        ttm = sec_screen.compute_ratios(ttm)

        with _cache_lock:
            _enriched_cache = ttm.copy()
            _cache_fingerprint = fingerprint
        return ttm
    finally:
        con.close()


def load_history_for_cik(cik: str) -> pd.DataFrame:
    require_pipeline()
    con = get_db_connection()
    try:
        if con.execute(
            "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = 'fundamentals_history'"
        ).fetchone()[0] == 0:
            return pd.DataFrame()
        history = con.execute(
            "SELECT * FROM fundamentals_history WHERE CAST(cik AS VARCHAR) = ? ORDER BY period DESC",
            [str(cik)],
        ).fetchdf()
        if history.empty:
            return history
        return sec_screen.compute_ratios(history)
    finally:
        con.close()


def _empty_summary() -> Dict[str, Any]:
    return {"total_cached": 0, "total_passed": 0, "avg_roic": None,
            "avg_operating_margin": None, "avg_debt_to_equity": None, "median_roic": None}


def get_ingest_summary() -> dict:
    """No-op-safe summary for the dashboard -- returns zeroed/empty
    values if fundamentals_history doesn't exist yet, never raises."""
    con = get_db_connection()
    try:
        if con.execute(
            "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = 'fundamentals_history'"
        ).fetchone()[0] == 0:
            return {"total_filings": 0, "total_companies": 0, "earliest_period": None,
                    "latest_period": None, "n_10k": 0, "n_10q": 0}
        row = con.execute("""
            SELECT COUNT(*), COUNT(DISTINCT cik), MIN(period), MAX(period),
                   SUM(CASE WHEN UPPER(form)='10-K' THEN 1 ELSE 0 END),
                   SUM(CASE WHEN UPPER(form)='10-Q' THEN 1 ELSE 0 END)
            FROM fundamentals_history
        """).fetchone()
        return {"total_filings": row[0], "total_companies": row[1], "earliest_period": row[2],
                "latest_period": row[3], "n_10k": row[4], "n_10q": row[5]}
    finally:
        con.close()


def get_ingest_log() -> list:
    """Returns the ingest log as a list of dicts, most recent first."""
    con = get_db_connection()
    try:
        if con.execute(
            "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = 'ingest_log'"
        ).fetchone()[0] == 0:
            return []
        rows = con.execute(
            "SELECT source_name, filings_added, ingested_at FROM ingest_log ORDER BY ingested_at DESC"
        ).fetchall()
        return [{"source_name": r[0], "filings_added": r[1], "ingested_at": r[2]} for r in rows]
    finally:
        con.close()


def screen_data_from_cache(params: Dict[str, Any], include_market_data: bool = False) -> Tuple[pd.DataFrame, ScreenInfo]:
    threshold_values = parse_thresholds(params)
    df = load_cached_ratios()
    if df.empty:
        return df, {"thresholds": threshold_values, "errors": [], "diagnostics": {"cached_rows": 0}, "summary": _empty_summary()}

    exchange_warning = None
    if threshold_values.get("major_exchanges_only"):
        df, exchange_warning = filter_to_major_exchanges(df)
        if df.empty:
            errors = [exchange_warning] if exchange_warning else []
            return df, {"thresholds": threshold_values, "errors": errors,
                        "diagnostics": {"cached_rows": 0}, "summary": _empty_summary()}

    diagnostics = {
        "cached_rows": len(df),
        "non_null_roic": int(df["ROIC"].notna().sum()) if "ROIC" in df.columns else 0,
        "passes_min_roic": int((df["ROIC"] > threshold_values["min_roic"]).sum()) if "ROIC" in df.columns else 0,
        "passes_min_operating_margin": int((df["OperatingMargin"] > threshold_values["min_operating_margin"]).sum()) if "OperatingMargin" in df.columns else 0,
        "passes_max_debt_to_equity": int((df["DebtToEquity"] < threshold_values["max_debt_to_equity"]).sum()) if "DebtToEquity" in df.columns else 0,
        "passes_min_interest_coverage": int((df["InterestCoverage"] > threshold_values["min_interest_coverage"]).sum()) if "InterestCoverage" in df.columns else 0,
        "passes_min_cfo_to_ni": int((df["CFO_to_NI"] > threshold_values["min_cfo_to_ni"]).sum()) if "CFO_to_NI" in df.columns else 0,
        "passes_min_fscore": int((df["FScore"] >= int(threshold_values.get("min_fscore", 0))).sum()) if "FScore" in df.columns else 0,
    }

    # --- Exclude financial sector (SIC 6000-6799) ---
    # Banks, insurers, and other financials have fundamentally different
    # balance sheets — NWC and Net PPE are not meaningful for them, so
    # ROIC is nonsensical.  Optional toggle; off by default.
    if threshold_values.get("exclude_financials"):
        sic_col = "sic" if "sic" in df.columns else "SIC"
        if sic_col in df.columns:
            df = df[
                ~df[sic_col].astype(float).between(6000, 6799, inclusive="both")
            ].copy()

    # --- Hydrate persisted prices before valuation ---
    # This lets intrinsic-value columns (Graham, DCF) and margin-of-safety
    # percentages compute from the last-saved price without needing a live
    # market-data refresh every time.
    df = _hydrate_prices_from_db(df)

    filtered = sec_screen.apply_quality_screen(
        df,
        min_roic=threshold_values["min_roic"],
        min_operating_margin=threshold_values["min_operating_margin"],
        max_debt_to_equity=threshold_values["max_debt_to_equity"],
        min_interest_coverage=threshold_values["min_interest_coverage"],
        min_cfo_to_ni=threshold_values["min_cfo_to_ni"],
        require_positive_ni=threshold_values["require_positive_ni"],
    )
    ranked = sec_screen.rank_magic_formula(filtered)

    ranked, ticker_warning = attach_tickers(ranked)

    # --- Server-side text search (searches ALL filtered companies, not just
    #      the current page — unlike the old client-side DOM filter) ---
    search_term = (params.get("search") or "").strip()
    if search_term:
        q = search_term.lower()
        # Match against Ticker, company name (case-insensitive substring),
        # or exact CIK match (bare digits).
        search_cik_col = "cik" if "cik" in ranked.columns else "CIK"
        name_col = "name" if "name" in ranked.columns else "Name"
        match_mask = pd.Series(False, index=ranked.index)
        if "Ticker" in ranked.columns:
            match_mask |= ranked["Ticker"].astype(str).str.lower().str.contains(q, na=False, regex=False)
        if name_col in ranked.columns:
            match_mask |= ranked[name_col].astype(str).str.lower().str.contains(q, na=False, regex=False)
        if search_cik_col in ranked.columns:
            match_mask |= ranked[search_cik_col].astype(str).str.strip() == q
        ranked = ranked[match_mask].copy()

    # Revenue growth filter (post-hoc, since it's computed after ratios)
    min_rev_growth = threshold_values.get("min_revenue_growth", 0.0)
    if "RevenueGrowth" in ranked.columns and min_rev_growth > -99:
        ranked = ranked[
            ranked["RevenueGrowth"].isna() | (ranked["RevenueGrowth"] >= min_rev_growth)
        ].copy()

    # F-Score filter (Piotroski 0-9).  Computed during enrichment so this is
    # also post-hoc, like revenue growth.  Defaults to 0 (no filter).
    min_fscore = int(threshold_values.get("min_fscore", 0))
    if min_fscore > 0 and "FScore" in ranked.columns:
        ranked = ranked[
            ranked["FScore"].isna() | (ranked["FScore"] >= min_fscore)
        ].copy()

    # Sector column from SIC
    from core.formatting import sic_to_sector
    sic_col = "sic" if "sic" in ranked.columns else "SIC"
    if sic_col in ranked.columns:
        ranked["Sector"] = ranked[sic_col].apply(sic_to_sector)

    # Sector-relative percentile ranks — a company with 15% ROIC might be
    # exceptional in Utilities but mediocre in Software.  Only computed when
    # a sector has ≥5 companies (percentiles are noise otherwise).
    for metric, out_col in [("ROIC", "ROIC_SectorPct"),
                             ("OperatingMargin", "OM_SectorPct")]:
        if metric in ranked.columns and "Sector" in ranked.columns:
            ranked[out_col] = ranked.groupby("Sector")[metric].transform(
                lambda x: x.rank(pct=True) * 100 if len(x) >= 5 else None
            )

    # Intrinsic value (Graham Number, DCF) — computable from fundamentals
    # alone, no price needed.
    ranked = apply_valuation(
        ranked,
        growth_rate=threshold_values["growth_rate"],
        discount_rate=threshold_values["discount_rate"],
        terminal_growth_rate=threshold_values["terminal_growth_rate"],
        projection_years=threshold_values["projection_years"],
    )

    # Margin of safety — needs a Price column.  Now that we hydrate persisted
    # prices above, this will produce results for any company whose price was
    # saved from a prior market-data refresh.
    if "Price" in ranked.columns and ranked["Price"].notna().any():
        ranked = compute_margin_of_safety(ranked)

    summary = {
        "total_cached": len(df),
        "total_passed": len(filtered),
        "avg_roic": float(df["ROIC"].mean()) if "ROIC" in df.columns and df["ROIC"].notna().any() else None,
        "avg_operating_margin": float(df["OperatingMargin"].mean()) if "OperatingMargin" in df.columns and df["OperatingMargin"].notna().any() else None,
        "avg_debt_to_equity": float(df["DebtToEquity"].replace([float("inf")], pd.NA).mean()) if "DebtToEquity" in df.columns and df["DebtToEquity"].notna().any() else None,
        "median_roic": float(df["ROIC"].median()) if "ROIC" in df.columns and df["ROIC"].notna().any() else None,
    }

    if include_market_data:
        ranked, errors = join_market_data(ranked)
        # Persist fetched prices so they survive page reloads
        _save_market_prices_to_db(ranked)
        ranked = compute_margin_of_safety(ranked)
    else:
        errors = []

    if ticker_warning:
        errors = list(errors) + [ticker_warning]
    if exchange_warning:
        errors = list(errors) + [exchange_warning]
    return ranked, {"thresholds": threshold_values, "errors": errors, "diagnostics": diagnostics, "summary": summary}


# ---------------------------------------------------------------------------
# Market-price persistence (DuckDB) — so prices survive page reloads and
# margin-of-safety computes without a live refresh every time.
# ---------------------------------------------------------------------------

def save_market_prices_to_db(df: pd.DataFrame) -> None:
    """Upsert market prices from a DataFrame (post-join_market_data) into
    the persistent market_prices table.  Uses a single DuckDB INSERT FROM
    SELECT so N rows become 1 round trip instead of N."""
    cik_col = "cik" if "cik" in df.columns else "CIK"
    if cik_col not in df.columns or "Price" not in df.columns:
        return

    # Build a clean DataFrame with only the columns we persist, handling NaN.
    persist_cols = {
        cik_col: "cik",
        "Ticker": "ticker",
        "Price": "price",
        "MarketCap": "market_cap",
        "EnterpriseValue": "enterprise_value",
        "EarningsYield": "earnings_yield",
        "PE": "pe",
        "PB": "pb",
        "EVToEBIT": "ev_to_ebit",
        "PFCF": "p_fcf",
        "MagicFormulaRank": "magic_formula_rank",
    }
    clean_rows = []
    for _, row in df.iterrows():
        cik = str(row.get(cik_col) or "").strip()
        if not cik:
            continue
        price = row.get("Price")
        if price is None or (isinstance(price, float) and math.isnan(price)):
            continue
        entry = {"cik": cik}
        for df_col, _db_col in persist_cols.items():
            if df_col == cik_col:
                continue
            val = row.get(df_col)
            if val is None or (isinstance(val, float) and math.isnan(val)):
                entry[_db_col] = None
            elif df_col == "Ticker":
                entry[_db_col] = str(val)
            elif df_col == "MagicFormulaRank":
                try:
                    entry[_db_col] = int(float(val))
                except (TypeError, ValueError):
                    entry[_db_col] = None
            else:
                try:
                    entry[_db_col] = float(val)
                except (TypeError, ValueError):
                    entry[_db_col] = None
        clean_rows.append(entry)

    if not clean_rows:
        return

    batch_df = pd.DataFrame(clean_rows)
    con = get_db_connection()
    try:
        con.execute("""
            CREATE TABLE IF NOT EXISTS market_prices (
                cik VARCHAR PRIMARY KEY,
                ticker VARCHAR,
                price DOUBLE,
                market_cap DOUBLE,
                enterprise_value DOUBLE,
                earnings_yield DOUBLE,
                pe DOUBLE,
                pb DOUBLE,
                ev_to_ebit DOUBLE,
                p_fcf DOUBLE,
                magic_formula_rank BIGINT,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        con.register("_mp_batch", batch_df)
        con.execute("""
            INSERT OR REPLACE INTO market_prices
                (cik, ticker, price, market_cap, enterprise_value,
                 earnings_yield, pe, pb, ev_to_ebit, p_fcf,
                 magic_formula_rank, last_updated)
            SELECT cik, ticker, price, market_cap, enterprise_value,
                   earnings_yield, pe, pb, ev_to_ebit, p_fcf,
                   magic_formula_rank, CURRENT_TIMESTAMP
            FROM _mp_batch
        """)
        con.unregister("_mp_batch")
        con.commit()
    finally:
        con.close()


# Backward-compatible alias so existing callers don't break.
_save_market_prices_to_db = save_market_prices_to_db


def _hydrate_prices_from_db(df: pd.DataFrame) -> pd.DataFrame:
    """Left-join persisted market prices from market_prices into df so
    margin-of-safety and multiples show up without a live refresh."""
    con = get_db_connection()
    try:
        if con.execute(
            "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = 'market_prices'"
        ).fetchone()[0] == 0:
            return df  # table doesn't exist yet — no persisted prices
        prices_df = con.execute("SELECT * FROM market_prices").fetchdf()
    finally:
        con.close()

    if prices_df.empty:
        return df

    df = df.copy()
    cik_col = "cik" if "cik" in df.columns else "CIK"
    prices_df["cik"] = prices_df["cik"].astype(str)
    df["_cik_str"] = df[cik_col].astype(str)

    price_cols = ["price", "market_cap", "enterprise_value", "earnings_yield",
                  "pe", "pb", "ev_to_ebit", "p_fcf", "magic_formula_rank", "ticker"]
    col_map = {
        "price": "Price", "market_cap": "MarketCap",
        "enterprise_value": "EnterpriseValue", "earnings_yield": "EarningsYield",
        "pe": "PE", "pb": "PB", "ev_to_ebit": "EVToEBIT", "p_fcf": "PFCF",
        "magic_formula_rank": "MagicFormulaRank",
    }

    for db_col, df_col in col_map.items():
        if db_col in prices_df.columns:
            lookup = prices_df.set_index("cik")[db_col].to_dict()
            df[df_col] = df["_cik_str"].map(lookup)

    # If Ticker wasn't set by attach_tickers, use persisted ticker
    if "ticker" in prices_df.columns and ("Ticker" not in df.columns or df["Ticker"].isna().all()):
        lookup = prices_df.set_index("cik")["ticker"].to_dict()
        df["Ticker"] = df["_cik_str"].map(lookup)

    df.drop(columns=["_cik_str"], inplace=True)
    return df


def paginate_frame(df: pd.DataFrame, params: Dict[str, Any]) -> Tuple[pd.DataFrame, Pagination]:
    sort_by = params.get("sort", "roic")
    order = params.get("order", "asc")
    page = max(1, int(params.get("page", 1)))
    sort_map = {
        "ticker": "Ticker",
        "cik": "cik" if "cik" in df.columns else "CIK",
        "name": "name" if "name" in df.columns else "Name",
        "sic": "sic" if "sic" in df.columns else "SIC",
        "roic": "ROIC",
        "operating_margin": "OperatingMargin",
        "debt_to_equity": "DebtToEquity",
        "interest_coverage": "InterestCoverage",
        "cfo_to_ni": "CFO_to_NI",
        "magic_formula_rank": "MagicFormulaRank",
        # Valuation
        "price": "Price",
        "graham_number": "GrahamNumber",
        "dcf_intrinsic_value": "DCFIntrinsicValue",
        "margin_of_safety_dcf": "MarginOfSafetyDCF",
        "margin_of_safety_graham": "MarginOfSafetyGraham",
        "revenue_growth": "RevenueGrowth",
        "fcf_growth": "FCFGrowth",
        "fscore": "FScore",
        "pe_ratio": "PE",
        "ev_ebitda": "EVToEBITDA",
        "sector": "Sector",
    }
    if sort_by in sort_map and sort_map[sort_by] in df.columns:
        df = df.sort_values(sort_map[sort_by], ascending=(order != "desc"), kind="mergesort")

    total_rows = len(df)
    total_pages = max(1, math.ceil(total_rows / PAGE_SIZE)) if total_rows else 1
    page = min(page, total_pages)
    start = (page - 1) * PAGE_SIZE
    stop = start + PAGE_SIZE
    page_df = df.iloc[start:stop].copy()
    return page_df, {"page": page, "total_pages": total_pages, "page_size": PAGE_SIZE, "total_rows": total_rows}


def coerce_row_values(row: pd.Series) -> Dict[str, Any]:
    values: Dict[str, Any] = {}
    for column in row.index:
        value = row[column]
        if isinstance(value, (float, int)):
            fv = float(value)
            if math.isnan(fv):
                values[column] = None
            else:
                values[column] = value
        else:
            values[column] = value
    return values


def get_display_rank(values: Dict[str, Any]) -> Tuple[Optional[Any], bool]:
    if values.get("MagicFormulaRank") is not None:
        return values.get("MagicFormulaRank"), True
    for key in ["roic_rank", "ROICRank"]:
        if key in values and values.get(key) is not None:
            return values.get(key), False
    return None, False


def build_row_context(row: pd.Series, params: Dict[str, Any]) -> RowContext:
    values = coerce_row_values(row)
    ticker = values.get("Ticker") or values.get("ticker")
    cik_value = values.get("CIK") or values.get("cik")
    display_ticker = ticker or (f"CIK {cik_value}" if cik_value else "—")
    rank_value, is_true_magic_formula = get_display_rank(values)

    # Dollar delta between price and DCF intrinsic value
    price = values.get("Price")
    dcf_iv = values.get("DCFIntrinsicValue")
    graham_iv = values.get("GrahamNumber")
    dcf_delta = None
    graham_delta = None
    if price is not None and dcf_iv is not None:
        try:
            dcf_delta = float(dcf_iv) - float(price)
        except (TypeError, ValueError):
            pass
    if price is not None and graham_iv is not None:
        try:
            graham_delta = float(graham_iv) - float(price)
        except (TypeError, ValueError):
            pass

    return {
        "row": values,
        "display_ticker": display_ticker,
        "cik": values.get("CIK") or values.get("cik") or "",
        "company_url": f"/company/{values.get('CIK') or values.get('cik') or ''}?{build_query_string(params)}",
        "magic_formula_rank": rank_value,
        "is_true_magic_formula": is_true_magic_formula,
        "roic": values.get("ROIC"),
        "operating_margin": values.get("OperatingMargin"),
        "debt_to_equity": values.get("DebtToEquity"),
        "interest_coverage": values.get("InterestCoverage"),
        "cfo_to_ni": values.get("CFO_to_NI"),
        # Price & intrinsic value
        "price": price,
        "graham_number": graham_iv,
        "dcf_intrinsic_value": dcf_iv,
        "dcf_delta": dcf_delta,
        "graham_delta": graham_delta,
        "margin_of_safety_graham": values.get("MarginOfSafetyGraham"),
        "margin_of_safety_dcf": values.get("MarginOfSafetyDCF"),
        # Growth
        "revenue_growth": values.get("RevenueGrowth"),
        "revenue_growth_3yr": values.get("RevenueGrowth3yr"),
        "fcf_growth": values.get("FCFGrowth"),
        # F-Score
        "fscore": values.get("FScore"),
        # Multiples
        "pe_ratio": values.get("PE"),
        "pb_ratio": values.get("PB"),
        "ev_to_ebit": values.get("EVToEBIT"),
        "p_fcf": values.get("PFCF"),
        # Sector
        "sector": values.get("Sector"),
        "roic_sector_pct": values.get("ROIC_SectorPct"),
        "om_sector_pct": values.get("OM_SectorPct"),
        # Trend indicators
        "roic_trend": _trend_arrow(values.get("ROIC"), values.get("PriorROIC")),
        "margin_trend": _trend_arrow(values.get("OperatingMargin"), values.get("PriorOperatingMargin")),
    }


def _trend_arrow(current, prior) -> str:
    """Returns ▲, ▼, or — comparing current vs prior period metric."""
    if current is None or prior is None:
        return ""
    try:
        if float(current) > float(prior):
            return "▲"
        elif float(current) < float(prior):
            return "▼"
    except (TypeError, ValueError):
        pass
    return "—"


# ---------------------------------------------------------------------------
# Watchlist (persisted in DuckDB)
# ---------------------------------------------------------------------------

def load_watchlist() -> set:
    """Returns the set of CIKs currently on the watchlist."""
    con = get_db_connection()
    try:
        con.execute("""
            CREATE TABLE IF NOT EXISTS watchlist (
                cik VARCHAR PRIMARY KEY,
                added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        _ensure_watchlist_columns(con)
        rows = con.execute("SELECT cik FROM watchlist").fetchall()
        return {str(r[0]) for r in rows}
    finally:
        con.close()


def _ensure_watchlist_columns(con) -> None:
    """Add columns that may not exist in older watchlist tables."""
    for col, col_type in [("added_price", "DOUBLE"), ("added_ticker", "VARCHAR")]:
        try:
            con.execute(f"ALTER TABLE watchlist ADD COLUMN IF NOT EXISTS {col} {col_type}")
        except Exception:
            pass  # DuckDB version too old for ADD COLUMN IF NOT EXISTS — ignore


def toggle_watchlist(cik: str) -> dict:
    """Add or remove a CIK from the watchlist. Returns new state with
    added_price when available so the UI can show entry vs current."""
    con = get_db_connection()
    try:
        con.execute("""
            CREATE TABLE IF NOT EXISTS watchlist (
                cik VARCHAR PRIMARY KEY,
                added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        _ensure_watchlist_columns(con)

        existing = con.execute(
            "SELECT COUNT(*) FROM watchlist WHERE cik = ?", [str(cik)]
        ).fetchone()[0]
        if existing:
            # Return added_at / added_price before deleting so the UI can show
            # the removal confirmation with context.
            row = con.execute(
                "SELECT added_at, added_price FROM watchlist WHERE cik = ?", [str(cik)]
            ).fetchone()
            con.execute("DELETE FROM watchlist WHERE cik = ?", [str(cik)])
            con.commit()
            result = {"cik": str(cik), "on_watchlist": False}
            if row and row[0] is not None:
                result["added_at"] = str(row[0])
            if row and len(row) > 1 and row[1] is not None:
                result["added_price"] = float(row[1])
            return result
        else:
            # Look up current price from persisted market data so we record
            # what the stock was trading at when the user added it.
            added_price = None
            added_ticker = None
            try:
                price_row = con.execute(
                    "SELECT price, ticker FROM market_prices WHERE cik = ?", [str(cik)]
                ).fetchone()
                if price_row:
                    added_price = price_row[0]
                    added_ticker = price_row[1]
            except Exception:
                pass  # market_prices table might not exist yet

            con.execute(
                "INSERT INTO watchlist (cik, added_price, added_ticker) VALUES (?, ?, ?)",
                [str(cik), added_price, added_ticker],
            )
            con.commit()
            result = {"cik": str(cik), "on_watchlist": True}
            if added_price is not None:
                result["added_price"] = float(added_price)
            return result
    finally:
        con.close()


def load_watchlist_data() -> pd.DataFrame:
    """Returns a DataFrame of watchlist companies with their current TTM ratios,
    plus watchlist metadata (added_at, added_price) and current market price so
    the UI can show entry vs current and the gain/loss delta."""
    watchlist_ciks = load_watchlist()
    if not watchlist_ciks:
        return pd.DataFrame()
    try:
        df = load_cached_ratios()
    except FileNotFoundError:
        return pd.DataFrame()
    cik_col = "cik" if "cik" in df.columns else "CIK"
    mask = df[cik_col].astype(str).isin(watchlist_ciks)
    result = df[mask].copy()

    # --- Join watchlist metadata (added_at, added_price, added_ticker) ---
    con = get_db_connection()
    try:
        _ensure_watchlist_columns(con)
        wl_rows = con.execute(
            "SELECT cik, added_at, added_price, added_ticker FROM watchlist"
        ).fetchall()
        wl_map = {str(r[0]): {"added_at": str(r[1]) if r[1] else None,
                               "added_price": float(r[2]) if len(r) > 2 and r[2] is not None else None,
                               "added_ticker": str(r[3]) if len(r) > 3 and r[3] else None}
                  for r in wl_rows}

        # Hydrate current price from market_prices (may be fresher than what's
        # already in the TTM DataFrame from _hydrate_prices_from_db)
        if "Price" not in result.columns or result["Price"].isna().all():
            try:
                mp_rows = con.execute(
                    "SELECT cik, price FROM market_prices"
                ).fetchall()
                mp_map = {str(r[0]): float(r[1]) for r in mp_rows if r[1] is not None}
                result["Price"] = result[cik_col].astype(str).map(mp_map)
            except Exception:
                pass
    finally:
        con.close()

    # Attach watchlist metadata columns
    result["_wl_meta"] = result[cik_col].astype(str).map(wl_map)
    result["AddedAt"] = result["_wl_meta"].apply(
        lambda m: m["added_at"] if m is not None else None)
    result["AddedPrice"] = result["_wl_meta"].apply(
        lambda m: m["added_price"] if m is not None else None)
    result["AddedTicker"] = result["_wl_meta"].apply(
        lambda m: m["added_ticker"] if m is not None else None)
    result.drop(columns=["_wl_meta"], inplace=True)

    return result