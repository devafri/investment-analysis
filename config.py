"""App-wide configuration."""

import os

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Schwab is the only market data provider (requires one-time OAuth bootstrap
# via providers.schwab.setup or the web UI at /setup).
MARKET_DATA_PROVIDER = "schwab"
