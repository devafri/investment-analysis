"""Single source of truth for core pipeline imports. Every other module
imports FROM HERE — exactly one place that knows whether the import
succeeded and one clear error message if it didn't."""

try:
    import core.fundamentals.sec_loader as sec_screen
    from providers.sec_lookup import (
        get_cik_ticker_map,
        get_cik_exchange_map,
    )
    from providers.schwab.market_data import fetch_price_data
except Exception as exc:  # pragma: no cover - depends on local environment
    sec_screen = None
    fetch_price_data = None
    get_cik_ticker_map = None
    get_cik_exchange_map = None
    IMPORT_ERROR = exc
    print(f"\n*** WARNING: could not import core pipeline modules ***")
    print(f"*** {type(exc).__name__}: {exc}")
    print(f"*** The app will start, but ingest/screen routes will fail. ***\n")
else:
    IMPORT_ERROR = None


def require_pipeline() -> None:
    if IMPORT_ERROR is not None:
        raise RuntimeError(
            f"Failed to import core pipeline modules: "
            f"{type(IMPORT_ERROR).__name__}: {IMPORT_ERROR}\n\n"
            f"Run: pip install -r requirements.txt"
        )
