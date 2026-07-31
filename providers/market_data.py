"""Joins live price/market-cap data (via yfinance) onto a screened
DataFrame, to complete the Magic Formula's Earnings Yield component --
SEC's fundamentals data has no stock price at all.
"""

import json
import math
import time
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as FutureTimeoutError
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from core.paths import MARKET_CACHE_PATH, TICKER_CACHE_PATH, ensure_cache_dir
from providers.pipeline_imports import fetch_price_data, get_cik_ticker_map
import config

TICKER_CACHE_MAX_AGE_SECONDS = 24 * 60 * 60  # refresh once a day at most
MIN_TICKER_ENTRIES = 100  # Fewer than this likely means a truncated/corrupt cache


def coerce_float(value: Any) -> Optional[float]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def load_ticker_map(force_refresh: bool = False) -> Tuple[Dict[str, str], Optional[str]]:
    """Returns (cik -> ticker dict, error_message_or_None). This is the FAST
    half of market data -- one bulk file download covering every ticker SEC
    knows about, unlike fetch_price_data (one live call per company). Cached
    to disk so it's not re-downloaded on every screen render. Fails soft:
    an unreachable SEC endpoint returns an empty map + an error message
    rather than breaking the whole screen.
    """
    ensure_cache_dir()
    if not force_refresh and TICKER_CACHE_PATH.exists():
        age = time.time() - TICKER_CACHE_PATH.stat().st_mtime
        if age < TICKER_CACHE_MAX_AGE_SECONDS:
            try:
                cached = json.loads(TICKER_CACHE_PATH.read_text())
                if len(cached) >= MIN_TICKER_ENTRIES:
                    return cached, None
                # else: cache looks truncated/corrupt -- fall through and refetch
            except (json.JSONDecodeError, OSError):
                pass  # fall through and refetch

    if get_cik_ticker_map is None:
        return {}, "Ticker map helper is not available in this environment."

    try:
        ticker_map = get_cik_ticker_map()
    except Exception as exc:  # pragma: no cover - depends on live network
        if TICKER_CACHE_PATH.exists():
            try:
                cached = json.loads(TICKER_CACHE_PATH.read_text())
                if len(cached) >= MIN_TICKER_ENTRIES:
                    return cached, f"Could not refresh ticker map ({exc}); using a previously cached copy."
            except (json.JSONDecodeError, OSError):
                pass
        return {}, f"Could not fetch SEC's ticker map: {exc}"

    if not ticker_map:
        return {}, "SEC's ticker map came back empty."

    try:
        TICKER_CACHE_PATH.write_text(json.dumps(ticker_map))
    except OSError:
        pass  # non-fatal -- just won't be cached for next time
    return ticker_map, None


def attach_tickers(df: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[str]]:
    """Populates a 'Ticker' column from the cached CIK->ticker map. This is
    deliberately separate from join_market_data (price/market cap): ticker
    resolution is one cheap bulk lookup and should show up immediately on
    the screen, not only after the slow "Refresh Market Data" step. If a
    CIK has no match (private/foreign issuer, or SEC's map hasn't been
    updated for it yet), the Ticker column is left null -- the UI falls back
    to showing "CIK <number>" rather than silently displaying a bare number
    that could be mistaken for a ticker.
    """
    if "Ticker" in df.columns and df["Ticker"].notna().any():
        return df, None  # already populated (e.g. after a market data refresh)

    ticker_map, error = load_ticker_map()
    df = df.copy()
    cik_col = "cik" if "cik" in df.columns else ("CIK" if "CIK" in df.columns else None)
    if cik_col is None or not ticker_map:
        df["Ticker"] = None
        return df, error

    df["Ticker"] = df[cik_col].apply(lambda cik: ticker_map.get(str(cik)))
    return df, error


def _apply_price_result(df: pd.DataFrame, index: Any, prices: Any, row: pd.Series) -> None:
    """Write one ticker's fetched price data into df at the given index.

    Fixes from the previous version:
    1. Raw price was fetched but never stored anywhere -- only used
       in-memory to derive market cap, then discarded. Now stored as its
       own 'Price' column (needed for margin-of-safety comparisons).
    2. 'EnterpriseValue' was previously set literally equal to MarketCap,
       with no debt/cash adjustment at all -- and 'EarningsYield' used Net
       Income divided by that, which is actually an inverse P/E, not
       Greenblatt's Magic Formula definition (EBIT / Enterprise Value).
       Both are now computed correctly: EV = MarketCap + TotalDebt - Cash,
       EarningsYield = OperatingIncomeLoss / EnterpriseValue.
    3. Not every price provider reliably supplies shares outstanding for
       every ticker (some asset types, delisted/thinly-traded names, or
       API hiccups can leave it null even when the provider generally
       supports it). Falls back to the SEC filing's own
       CommonStockSharesOutstanding (already used elsewhere for EPS/Book
       Value) in that case, so MarketCap/Enterprise Value/Earnings Yield
       can still be computed rather than going permanently blank.
    """
    if not isinstance(prices, dict):
        return

    price = prices.get("price")
    market_cap = prices.get("market_cap")
    if market_cap is None:
        shares_out = prices.get("shares_out")
        if shares_out is None:
            # Provider didn't supply a share count -- use the SEC filing's
            # own figure instead of leaving market cap unavailable entirely.
            shares_out = row.get("CommonStockSharesOutstanding")
        if price is not None and shares_out is not None:
            try:
                market_cap = float(price) * float(shares_out)
            except (TypeError, ValueError):
                market_cap = None

    df.at[index, "Price"] = price
    df.at[index, "MarketCap"] = market_cap

    if market_cap in (None, ""):
        return

    total_debt = row.get("TotalDebt")
    cash = row.get("CashAndCashEquivalents")
    try:
        enterprise_value = float(market_cap) + float(total_debt or 0) - float(cash or 0)
    except (TypeError, ValueError):
        enterprise_value = None
    df.at[index, "EnterpriseValue"] = enterprise_value

    ebit = row.get("OperatingIncomeLoss")
    if enterprise_value not in (None, "") and ebit not in (None, ""):
        try:
            if float(enterprise_value) != 0:
                df.at[index, "EarningsYield"] = float(ebit) / float(enterprise_value)
        except (TypeError, ValueError):
            df.at[index, "EarningsYield"] = None

    # --- Standard valuation multiples ---
    shares = row.get("CommonStockSharesOutstanding")
    ni = row.get("NetIncomeLoss")
    equity = row.get("StockholdersEquity")
    fcf = row.get("FCF")

    try:
        _s = float(shares) if shares is not None and not (isinstance(shares, float) and math.isnan(shares)) else None
    except (TypeError, ValueError):
        _s = None

    # P/E
    if price is not None and _s is not None and _s > 0 and ni is not None:
        try:
            eps = float(ni) / _s
            if eps > 0:
                df.at[index, "PE"] = float(price) / eps
        except (TypeError, ValueError, ZeroDivisionError):
            pass

    # P/B
    if price is not None and _s is not None and _s > 0 and equity is not None:
        try:
            bvps = float(equity) / _s
            if bvps > 0:
                df.at[index, "PB"] = float(price) / bvps
        except (TypeError, ValueError, ZeroDivisionError):
            pass

    # EV/EBIT
    ebit = row.get("OperatingIncomeLoss")
    if enterprise_value not in (None, "") and ebit not in (None, ""):
        try:
            if float(ebit) > 0:
                df.at[index, "EVToEBIT"] = float(enterprise_value) / float(ebit)
        except (TypeError, ValueError, ZeroDivisionError):
            pass

    # P/FCF
    if market_cap not in (None, "") and fcf is not None:
        try:
            if float(fcf) > 0:
                df.at[index, "PFCF"] = float(market_cap) / float(fcf)
        except (TypeError, ValueError, ZeroDivisionError):
            pass


def join_market_data(df: pd.DataFrame, overall_timeout_seconds: float = 25.0) -> Tuple[pd.DataFrame, List[str]]:
    """Join live price/market-cap data onto df, via whichever provider is
    configured in config.MARKET_DATA_PROVIDER ("yfinance", the default, or
    "schwab" once you've completed schwab_setup.py).

    yfinance path: one live network call PER ticker (no bulk endpoint
    exists), so callers should pass in only the rows they actually need
    enriched (e.g. the current page of ~20 results), not an entire filtered
    universe of hundreds of companies. Fetches run concurrently (bounded
    thread pool) with a hard wall-clock deadline for the whole batch; any
    ticker that hasn't responded by the deadline is recorded as an error and
    skipped. The pool is shut down with wait=False so a still-running
    stalled thread doesn't block this request from returning -- using
    `with ThreadPoolExecutor(...) as pool:` here would defeat the timeout
    entirely, since the context manager's __exit__ blocks until every
    submitted thread finishes, no matter how long that takes.

    schwab path: ONE batch API call for all tickers at once, which is both
    faster and avoids the per-ticker rate-limiting problems yfinance is
    known for -- this is the real advantage of having Schwab access.
    """
    if get_cik_ticker_map is None:
        return df, ["Market data helpers are not available in this environment."]

    df = df.copy()
    df["Ticker"] = None
    df["Price"] = None
    df["MarketCap"] = None
    df["EnterpriseValue"] = None
    df["EarningsYield"] = None
    df["MagicFormulaRank"] = None

    try:
        cik_to_ticker = get_cik_ticker_map() or {}
    except Exception as exc:  # pragma: no cover - depends on live network
        return df, [f"Could not download the SEC CIK<->ticker map: {exc}"]

    errors: List[str] = []

    # Resolve which rows have a ticker at all, once, regardless of provider.
    index_to_ticker: Dict[Any, str] = {}
    for index, row in df.iterrows():
        cik = str(row.get("CIK") or row.get("cik") or "").strip()
        if not cik:
            continue
        ticker = cik_to_ticker.get(cik)
        if not ticker:
            continue
        df.at[index, "Ticker"] = ticker
        index_to_ticker[index] = ticker

    if not index_to_ticker:
        return df, errors

    if config.MARKET_DATA_PROVIDER == "schwab":
        errors.extend(_fetch_via_schwab(df, index_to_ticker))
    else:
        errors.extend(_fetch_via_yfinance(df, index_to_ticker, overall_timeout_seconds))

    ranked_indices = df.index[df["EarningsYield"].notna()].tolist()
    ranked_indices = sorted(
        ranked_indices,
        key=lambda idx: (
            -float(df.at[idx, "EarningsYield"]),
            -float(df.at[idx, "ROIC"]) if coerce_float(df.at[idx, "ROIC"]) is not None else 0,
        ),
    )
    for rank, idx in enumerate(ranked_indices, start=1):
        df.at[idx, "MagicFormulaRank"] = rank

    if errors:
        MARKET_CACHE_PATH.write_text(json.dumps({"errors": errors}), encoding="utf-8")
    return df, errors


def _fetch_via_yfinance(df: pd.DataFrame, index_to_ticker: Dict[Any, str], overall_timeout_seconds: float) -> List[str]:
    """The original per-ticker, thread-pool-with-deadline approach. Kept as
    its own function so join_market_data's provider branching stays readable."""
    if fetch_price_data is None:
        return ["yfinance helper is not available in this environment."]

    errors: List[str] = []
    pool = ThreadPoolExecutor(max_workers=8)
    future_to_meta: Dict[Any, Tuple[Any, str]] = {}
    try:
        for index, ticker in index_to_ticker.items():
            future = pool.submit(fetch_price_data, ticker)
            future_to_meta[future] = (index, ticker)

        try:
            completed = as_completed(future_to_meta, timeout=overall_timeout_seconds)
            for future in completed:
                index, ticker = future_to_meta[future]
                try:
                    prices = future.result()
                except Exception as exc:  # pragma: no cover - depends on live network
                    row = df.loc[index] if index in df.index else None
                    cik = (row.get("CIK") or row.get("cik") or "?") if row is not None else "?"
                    errors.append(f"{ticker} [CIK {cik}]: {exc}")
                    continue
                _apply_price_result(df, index, prices, row=df.loc[index])
        except FutureTimeoutError:
            pending = [ticker for f, (_, ticker) in future_to_meta.items() if not f.done()]
            if pending:
                shown = ", ".join(pending[:10])
                more = f" (+{len(pending) - 10} more)" if len(pending) > 10 else ""
                errors.append(
                    f"{len(pending)} ticker(s) did not respond within {overall_timeout_seconds:.0f}s "
                    f"and were skipped: {shown}{more}"
                )
    finally:
        pool.shutdown(wait=False)
    return errors


def _fetch_via_schwab(df: pd.DataFrame, index_to_ticker: Dict[Any, str]) -> List[str]:
    """One batch API call for every ticker, instead of yfinance's
    one-call-per-ticker. Requires schwab_setup.py to have been run already."""
    try:
        from providers.schwab.market_data import fetch_prices_batch
        from providers.schwab.auth import SchwabAuthError
    except ImportError as exc:
        return [f"Schwab integration not available: {exc}. Falling back would require "
                f"restarting with MARKET_DATA_PROVIDER=yfinance."]

    tickers = list(index_to_ticker.values())
    try:
        results = fetch_prices_batch(tickers)
    except SchwabAuthError as exc:
        return [f"Schwab authentication problem: {exc}"]
    except Exception as exc:  # pragma: no cover - depends on live network
        return [f"Schwab quotes request failed: {exc}"]

    errors: List[str] = []
    for index, ticker in index_to_ticker.items():
        prices = results.get(ticker)
        if prices is None:
            row = df.loc[index]
            cik = row.get("CIK") or row.get("cik") or "?"
            name = row.get("Name") or row.get("name") or ""
            name_suffix = f" ({name})" if name else ""
            errors.append(
                f"{ticker}{name_suffix} [CIK {cik}]: no quote returned by Schwab "
                f"(preferred share, delisted, or not in Schwab's universe)"
            )
            continue
        _apply_price_result(df, index, prices, row=df.loc[index])
    return errors