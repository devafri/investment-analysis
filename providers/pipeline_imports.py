"""Single source of truth for importing the two pipeline scripts
(sec_value_screen.py, fetch_market_data.py). Every other module imports
FROM HERE rather than doing its own try/except -- so there's exactly one
place that knows whether the import succeeded, and exactly one clear error
message if it didn't, instead of the failure mode being duplicated (and
potentially worded differently) in five different files.
"""

try:
    import core.sec_value_screen as sec_screen
    from providers.yfinance_provider import fetch_price_data, get_cik_ticker_map, get_cik_exchange_map
except Exception as exc:  # pragma: no cover - depends on local environment
    sec_screen = None
    fetch_price_data = None
    get_cik_ticker_map = None
    get_cik_exchange_map = None
    IMPORT_ERROR = exc
    print(f"\n*** WARNING: could not import core/sec_value_screen.py / providers/yfinance_provider.py ***")
    print(f"*** {type(exc).__name__}: {exc}")
    print(f"*** The app will start, but ingest/screen routes will fail until this is fixed. ***\n")
else:
    IMPORT_ERROR = None


def require_pipeline() -> None:
    if IMPORT_ERROR is not None:
        raise RuntimeError(
            f"Failed to import core/sec_value_screen.py / providers/yfinance_provider.py: "
            f"{type(IMPORT_ERROR).__name__}: {IMPORT_ERROR}\n\n"
            f"If this says 'No module named X', run: pip install -r requirements.txt\n"
            f"If this says 'No such file or directory' or similar, make sure "
            f"core/sec_value_screen.py and providers/yfinance_provider.py are present."
        )