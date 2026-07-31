"""Shared filesystem paths used across the app. Kept in one place so every
module agrees on where the cache/DB/templates live, instead of each module
recomputing Path(__file__).resolve().parent independently."""

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent  # repo root (core/paths.py → core/ → repo/)
TEMPLATES_DIR = BASE_DIR / "templates"
CACHE_DIR = BASE_DIR / "cache"
DB_PATH = CACHE_DIR / "screen.duckdb"
MARKET_CACHE_PATH = CACHE_DIR / "market_data.json"
EXCHANGE_CACHE_PATH = CACHE_DIR / "exchange_map.json"
TICKER_CACHE_PATH = CACHE_DIR / "ticker_map.json"


def ensure_cache_dir() -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)