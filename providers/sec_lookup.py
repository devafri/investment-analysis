"""SEC public endpoints for CIK↔ticker and CIK↔exchange mapping.

These are free, unauthenticated SEC REST endpoints — NOT dependent on
yfinance or Schwab.  Previously bundled inside yfinance_provider.py.
"""

import requests

SEC_TICKER_URL = "https://www.sec.gov/files/company_tickers.json"
SEC_EXCHANGE_URL = "https://www.sec.gov/files/company_tickers_exchange.json"
HEADERS = {"User-Agent": "Research script contact@example.com"}


def get_cik_ticker_map() -> dict:
    """Return {cik_str: ticker} from SEC's company_tickers.json."""
    r = requests.get(SEC_TICKER_URL, headers=HEADERS, timeout=30)
    r.raise_for_status()
    data = r.json()
    return {str(v["cik_str"]): v["ticker"] for v in data.values()}


def get_cik_exchange_map() -> dict:
    """Return {cik_str: exchange} from SEC's company_tickers_exchange.json."""
    r = requests.get(SEC_EXCHANGE_URL, headers=HEADERS, timeout=30)
    r.raise_for_status()
    data = r.json()
    fields = data.get("fields", [])
    rows = data.get("data", [])
    if not fields or not rows:
        return {}
    cik_idx = fields.index("cik") if "cik" in fields else 0
    exch_idx = fields.index("exchange") if "exchange" in fields else len(fields) - 1
    result = {}
    for row in rows:
        try:
            cik = str(row[cik_idx])
            exch = row[exch_idx]
        except (IndexError, TypeError):
            continue
        if exch:
            result[cik] = exch
    return result
