"""App-wide configuration. Currently just the market data provider choice --
kept as an environment variable (not a UI toggle) since switching providers
is a one-time setup decision (Schwab requires the OAuth bootstrap in
schwab_setup.py first), not something to flip per-screen-load.
"""

import os

try:
    from dotenv import load_dotenv
    load_dotenv()  # loads .env into os.environ if present; harmless no-op otherwise.
except ImportError:
    pass  # python-dotenv not installed -- fall back to whatever's already in the environment

# "schwab" (default -- richer data: real shares outstanding/market cap plus
# a battery of pre-computed ratios, and one batch call instead of one per
# ticker, but requires schwab_setup.py's one-time OAuth step first) or
# "yfinance" (zero setup, but less reliable/rate-limited, and has no
# shares-outstanding/market-cap data of its own).
MARKET_DATA_PROVIDER = os.environ.get("MARKET_DATA_PROVIDER", "schwab").strip().lower()