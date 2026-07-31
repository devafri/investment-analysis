#!/usr/bin/env python3
"""
fetch_market_data.py
---------------------
Companion to sec_value_screen.py. The SEC DERA fundamentals data has no stock
price, so Greenblatt's Earnings Yield (EBIT / Enterprise Value) can't be
computed from SEC data alone. This script joins in price + shares outstanding
via yfinance (requires internet + a ticker list) and finishes the ranking.

You need a CIK -> ticker mapping since SEC data is keyed by CIK, not ticker.
SEC publishes one for free (no auth needed):
    https://www.sec.gov/files/company_tickers.json

USAGE
    pip install yfinance requests pandas
    python3 fetch_market_data.py --screen full_test.csv --out ranked_final.csv

WHAT IT DOES
1. Downloads the SEC's CIK<->ticker mapping.
2. For each company in your screen output, looks up its ticker, then pulls
   current price and shares outstanding via yfinance.
3. Computes:
   MarketCap        = Price * SharesOutstanding
   EnterpriseValue   = MarketCap + TotalDebt - Cash
   EarningsYield     = EBIT / EnterpriseValue
4. Combines ranks: Magic Formula Rank = rank(ROIC) + rank(EarningsYield),
   lower combined rank = more attractive (Greenblatt's method).

NOTE ON DATA FRESHNESS
Price is live (today), fundamentals are as-of the filing period -- there's an
inherent lag. That's normal for this kind of screen (Greenblatt's original
methodology has the same property) but worth remembering when interpreting
results a few quarters after the filing date.
"""

import argparse
import json
import time
import pandas as pd
import requests

SEC_TICKER_URL = "https://www.sec.gov/files/company_tickers.json"
HEADERS = {"User-Agent": "Research script contact@example.com"}  # SEC asks for a UA with contact info


def get_cik_ticker_map() -> dict:
    r = requests.get(SEC_TICKER_URL, headers=HEADERS, timeout=30)
    r.raise_for_status()
    data = r.json()
    # data is {"0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."}, ...}
    return {str(v["cik_str"]): v["ticker"] for v in data.values()}


SEC_EXCHANGE_URL = "https://www.sec.gov/files/company_tickers_exchange.json"


def get_cik_exchange_map() -> dict:
    """Returns {cik_str: exchange_name} for every ticker SEC has a listing
    exchange on file for (e.g. "Nasdaq", "NYSE", "NYSE Arca", "NYSE American",
    "CBOE", "OTC", ...). Used to filter the screen to only exchange-listed
    companies. SEC's field is free-text and not perfectly standardized across
    the exchange landscape -- callers should match on substrings (e.g.
    "nasdaq" in exchange.lower()) rather than exact equality, and should
    expect some legitimate NYSE/Nasdaq companies to be missing if SEC hasn't
    updated this particular file for them yet.
    """
    r = requests.get(SEC_EXCHANGE_URL, headers=HEADERS, timeout=30)
    r.raise_for_status()
    data = r.json()
    # Expected shape: {"fields": ["cik","name","ticker","exchange"], "data": [[320193,"Apple Inc.","AAPL","Nasdaq"], ...]}
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


def fetch_price_data(ticker: str) -> dict:
    import yfinance as yf
    try:
        t = yf.Ticker(ticker)
        info = t.fast_info
        return {
            "ticker": ticker,
            "price": info.get("lastPrice"),
            "shares_out": info.get("shares"),
            "market_cap": info.get("marketCap"),
        }
    except Exception as e:
        return {"ticker": ticker, "price": None, "shares_out": None, "market_cap": None, "error": str(e)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--screen", required=True, help="CSV from sec_value_screen.py (use --full-out for the pre-screen version)")
    ap.add_argument("--out", default="ranked_final.csv")
    ap.add_argument("--sleep", type=float, default=0.3, help="Delay between yfinance calls to be polite")
    args = ap.parse_args()

    df = pd.read_csv(args.screen)
    df["cik"] = df["cik"].astype(str)

    print("Fetching SEC CIK<->ticker map ...")
    cik_map = get_cik_ticker_map()
    df["ticker"] = df["cik"].map(cik_map)

    missing = df["ticker"].isna().sum()
    if missing:
        print(f"WARNING: {missing} companies had no ticker match (CIK not in SEC's public map, or foreign private issuer).")

    records = []
    for _, row in df.dropna(subset=["ticker"]).iterrows():
        print(f"  fetching {row['ticker']} ...")
        records.append(fetch_price_data(row["ticker"]))
        time.sleep(args.sleep)

    market_df = pd.DataFrame(records)
    merged = df.merge(market_df, on="ticker", how="left")

    merged["MarketCap"] = merged["market_cap"]
    merged["TotalDebt"] = merged.get("TotalDebt", 0)
    merged["EnterpriseValue"] = merged["MarketCap"] + merged.get("TotalDebt", 0).fillna(0) - merged.get("CashAndCashEquivalents", 0).fillna(0)
    merged["EarningsYield"] = merged["OperatingIncomeLoss"] / merged["EnterpriseValue"]

    merged["roic_rank"] = merged["ROIC"].rank(ascending=False)
    merged["ey_rank"] = merged["EarningsYield"].rank(ascending=False)
    merged["magic_formula_rank"] = merged["roic_rank"] + merged["ey_rank"]
    merged = merged.sort_values("magic_formula_rank")

    merged.to_csv(args.out, index=False)
    print(f"\nDone. Full Magic Formula ranking written to {args.out}")


if __name__ == "__main__":
    main()