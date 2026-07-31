"""Filters companies to NASDAQ/NYSE-listed only, using SEC's free
CIK-to-exchange listing (company_tickers_exchange.json). SEC's DERA
fundamentals data has no exchange info at all, so this is a separate lookup,
cached to disk since it's a several-MB download covering every listed
ticker.
"""

import json
import time
from typing import Any, Dict, Optional, Tuple

import pandas as pd

from core.paths import EXCHANGE_CACHE_PATH, ensure_cache_dir
from providers.pipeline_imports import get_cik_exchange_map

EXCHANGE_CACHE_MAX_AGE_SECONDS = 24 * 60 * 60  # refresh once a day at most
MIN_EXCHANGE_ENTRIES = 100  # Fewer than this likely means a truncated/corrupt cache

# Substring match (case-insensitive) against SEC's free-text exchange field --
# not an exact-equality set, since SEC's values aren't perfectly standardized
# (e.g. "Nasdaq", "NASDAQ Global Select", "New York Stock Exchange" all show up).
ALLOWED_EXCHANGE_SUBSTRINGS = ["nasdaq", "nyse", "new york stock exchange"]


def load_exchange_map(force_refresh: bool = False) -> Tuple[Dict[str, str], Optional[str]]:
    """Returns (cik -> exchange dict, error_message_or_None). Refreshes
    automatically once the cache is more than a day old, or immediately if
    force_refresh is set.

    Fails soft: if the fetch fails (no internet, SEC endpoint unreachable,
    unexpected response shape), returns an empty map + an error message
    rather than raising -- the caller decides whether to skip the exchange
    filter or surface the error, but a network hiccup here shouldn't take
    down the whole screen.
    """
    ensure_cache_dir()
    if not force_refresh and EXCHANGE_CACHE_PATH.exists():
        age = time.time() - EXCHANGE_CACHE_PATH.stat().st_mtime
        if age < EXCHANGE_CACHE_MAX_AGE_SECONDS:
            try:
                cached = json.loads(EXCHANGE_CACHE_PATH.read_text())
                if len(cached) >= MIN_EXCHANGE_ENTRIES:
                    return cached, None
                # else: cache looks truncated/corrupt -- fall through and refetch
            except (json.JSONDecodeError, OSError):
                pass  # fall through and refetch

    if get_cik_exchange_map is None:
        return {}, "Exchange listing helper is not available in this environment."

    try:
        exchange_map = get_cik_exchange_map()
    except Exception as exc:  # pragma: no cover - depends on live network
        # If we have a stale cache, prefer using it over failing outright.
        if EXCHANGE_CACHE_PATH.exists():
            try:
                cached = json.loads(EXCHANGE_CACHE_PATH.read_text())
                if len(cached) >= MIN_EXCHANGE_ENTRIES:
                    return cached, f"Could not refresh exchange listing ({exc}); using a previously cached copy."
            except (json.JSONDecodeError, OSError):
                pass
        return {}, f"Could not fetch SEC's exchange listing: {exc}"

    if not exchange_map:
        return {}, "SEC's exchange listing came back empty -- check https://www.sec.gov/files/company_tickers_exchange.json is reachable and has the expected format."

    try:
        EXCHANGE_CACHE_PATH.write_text(json.dumps(exchange_map))
    except OSError:
        pass  # non-fatal -- just won't be cached for next time
    return exchange_map, None


def is_major_exchange(exchange_name: Any) -> bool:
    if not exchange_name:
        return False
    exch_lower = str(exchange_name).lower()
    return any(s in exch_lower for s in ALLOWED_EXCHANGE_SUBSTRINGS)


def filter_to_major_exchanges(df: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[str]]:
    """Filters df to only rows whose CIK maps to NASDAQ or NYSE per SEC's
    exchange listing. Returns (filtered_df, warning_message_or_None) -- if
    the exchange map can't be loaded, returns df UNFILTERED with a warning
    rather than silently emptying the whole screen."""
    exchange_map, error = load_exchange_map()
    if not exchange_map:
        return df, (error or "No exchange listing data available -- showing all companies unfiltered.")

    cik_col = "cik" if "cik" in df.columns else ("CIK" if "CIK" in df.columns else None)
    if cik_col is None:
        return df, "Could not find a CIK column to filter by exchange."

    mask = df[cik_col].apply(lambda cik: is_major_exchange(exchange_map.get(str(cik))))
    return df[mask].copy(), None


def get_allowed_cik_set(exchange_map: Dict[str, str]) -> set:
    """Used at ingest time to pre-filter companies before the heavy
    num.txt join/pivot step -- see sec_value_screen.load_data_filtered."""
    return {cik for cik, exch in exchange_map.items() if is_major_exchange(exch)}