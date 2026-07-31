"""Price/quote fetching via Schwab's Market Data API -- an alternative to
fetch_market_data.py's yfinance-based fetch_price_data, for anyone with
Schwab API access.

IMPORTANT, READ BEFORE DEBUGGING: I cannot verify the exact response shape
of Schwab's /marketdata/v1/quotes endpoint from my environment -- no
internet access, and even with it, I don't have your credentials. The
parsing below is my best understanding of the (TD Ameritrade-derived) shape,
but Schwab may have changed field names or nesting since. If parsing fails,
`fetch_price_data` will raise with the RAW response included in the error --
read that, tell me what the actual shape looks like, and I'll fix the
parsing precisely rather than guessing again.
"""

from typing import Dict, List, Optional

import requests

from providers.schwab.auth import get_valid_access_token, SchwabAuthError

QUOTES_URL = "https://api.schwabapi.com/marketdata/v1/quotes"


def _format_error_response(resp: requests.Response) -> str:
    """Schwab's documented error shape is {"errors": [{"status", "title",
    "detail", "source": {...}}, ...]} -- extract the useful parts for a
    cleaner message instead of dumping raw response text. Falls back to raw
    text if the response doesn't parse as that shape (e.g. a proxy/gateway
    error that never reached Schwab's own error handling).
    """
    try:
        body = resp.json()
        errors = body.get("errors", [])
        if errors:
            parts = []
            for err in errors:
                detail = err.get("detail", "")
                title = err.get("title", "")
                source = err.get("source", {})
                source_str = f" (source: {source})" if source else ""
                parts.append(f"{title}: {detail}{source_str}" if title or detail else str(err))
            return f"Schwab quotes request failed (HTTP {resp.status_code}): " + "; ".join(parts)
    except (ValueError, AttributeError):
        pass
    return f"Schwab quotes request failed (HTTP {resp.status_code}): {resp.text}"


class SchwabQuoteError(Exception):
    """Raised when a quote request fails or comes back in an unexpected
    shape. Always includes enough of the raw response to debug from."""
    pass


def _parse_quote_entry(ticker: str, entry: dict) -> Dict[str, Optional[float]]:
    """Extraction of price/shares/market-cap/book-value from one ticker's
    quote entry.

    CORRECTED from an earlier version: the first documentation example I
    was shown only illustrated a handful of populated `fundamental` fields
    (eps, peRatio, divYield, avg volumes), which I read as "these are the
    only fields available." The full FundamentalInst schema shows that's
    wrong -- fundamental also includes sharesOutstanding, marketCap,
    marketCapFloat, and bookValuePerShare directly, plus a long list of
    ratios (epsTTM, returnOnEquity, returnOnAssets, currentRatio,
    quickRatio, interestCoverage, totalDebtToEquity, grossMarginTTM,
    operatingMarginTTM, netProfitMarginTTM) that overlap with what this app
    computes independently from SEC filings -- worth treating as a
    cross-check source later, not just a price feed.
    """
    quote = entry.get("quote", {})
    regular = entry.get("regular", {})
    fundamental = entry.get("fundamental", {})

    price = None
    for key in ("lastPrice", "mark", "closePrice"):
        if quote.get(key) is not None:
            price = quote[key]
            break
    if price is None and regular.get("regularMarketLastPrice") is not None:
        price = regular["regularMarketLastPrice"]

    if price is None:
        raise SchwabQuoteError(
            f"Could not find a price field for {ticker} in Schwab's response. "
            f"Raw entry: {entry}\n"
            f"Tell me what field actually has the price and I'll fix the parsing."
        )

    return {
        "ticker": ticker,
        "price": price,
        "shares_out": fundamental.get("sharesOutstanding"),
        "market_cap": fundamental.get("marketCap"),
        # Bonus fields beyond the shared price-provider interface -- not
        # consumed by market_data.py's _apply_price_result yet, but
        # available here if/when a cross-check against SEC-derived ratios
        # gets built.
        "book_value_per_share": fundamental.get("bookValuePerShare"),
        "eps_ttm": fundamental.get("epsTTM"),
        "pe_ratio": fundamental.get("peRatio"),
        "pb_ratio": fundamental.get("pbRatio"),
        "return_on_equity": fundamental.get("returnOnEquity"),
        "return_on_assets": fundamental.get("returnOnAssets"),
        "current_ratio": fundamental.get("currentRatio"),
        "quick_ratio": fundamental.get("quickRatio"),
        "interest_coverage": fundamental.get("interestCoverage"),
        "total_debt_to_equity": fundamental.get("totalDebtToEquity"),
        "gross_margin_ttm": fundamental.get("grossMarginTTM"),
        "operating_margin_ttm": fundamental.get("operatingMarginTTM"),
        "net_profit_margin_ttm": fundamental.get("netProfitMarginTTM"),
    }


def fetch_prices_batch(tickers: List[str]) -> Dict[str, dict]:
    """Fetch quotes for MULTIPLE tickers in one API call. This is the real
    advantage over yfinance's one-call-per-ticker approach: a batch of, say,
    20 tickers becomes ONE request instead of 20, which is both faster and
    avoids the per-ticker rate-limiting problems yfinance is known for.

    Returns {ticker: {"ticker", "price", "shares_out", "market_cap"}} for
    every ticker that resolved successfully. Tickers that failed to parse
    are simply omitted (caller can detect missing tickers by checking which
    keys are absent) rather than raising and losing the whole batch.
    """
    if not tickers:
        return {}

    access_token = get_valid_access_token()
    resp = requests.get(
        QUOTES_URL,
        headers={"Authorization": f"Bearer {access_token}"},
        params={"symbols": ",".join(tickers), "fields": "quote,fundamental,reference,regular"},
        timeout=30,
    )
    if not resp.ok:
        raise SchwabQuoteError(_format_error_response(resp))

    data = resp.json()
    results: Dict[str, dict] = {}
    for ticker in tickers:
        entry = data.get(ticker)
        if entry is None:
            continue  # ticker not found in response -- caller sees it's simply missing
        try:
            results[ticker] = _parse_quote_entry(ticker, entry)
        except SchwabQuoteError:
            continue  # skip this one, keep the rest of the batch
    return results


def fetch_price_data(ticker: str) -> dict:
    """Single-ticker interface matching fetch_market_data.fetch_price_data's
    signature, so this module is a drop-in replacement wherever that
    function is imported. Internally just calls the batch function with one
    ticker -- prefer fetch_prices_batch directly when fetching many tickers,
    since that's where Schwab's real advantage over yfinance shows up.
    """
    try:
        results = fetch_prices_batch([ticker])
    except (SchwabQuoteError, SchwabAuthError) as e:
        return {"ticker": ticker, "price": None, "shares_out": None, "market_cap": None, "error": str(e)}
    if ticker not in results:
        return {"ticker": ticker, "price": None, "shares_out": None, "market_cap": None, "error": "not found in Schwab response"}
    return results[ticker]