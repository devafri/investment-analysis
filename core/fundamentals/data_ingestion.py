"""Resolving user-provided data-dir/zip paths into actual folders containing
sub.txt/num.txt, and the shared DuckDB connection helper.
"""

import shutil
import zipfile
from pathlib import Path
from typing import List

import duckdb

from core.paths import CACHE_DIR, DB_PATH, ensure_cache_dir


def resolve_data_sources(data_dir: str) -> List[Path]:
    """Returns a list of resolved data folders (each containing sub.txt/num.txt),
    one per quarterly zip found. Each zip is extracted into its OWN subfolder
    (keyed by filename) rather than all being extracted into the same
    location -- extracting multiple zips into one shared folder would let
    later zips' sub.txt/num.txt silently overwrite earlier ones, which
    previously meant only the last quarter in a folder ever actually got
    used. Now every zip is ingested and accumulated into history."""
    data_path = Path(data_dir).expanduser()

    if data_path.is_file() and data_path.suffix.lower() == ".zip":
        extract_root = CACHE_DIR / "extracted" / data_path.stem
        if extract_root.exists():
            shutil.rmtree(extract_root)
        extract_root.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(data_path) as archive:
            archive.extractall(extract_root)
        if (extract_root / "sub.txt").exists() and (extract_root / "num.txt").exists():
            return [extract_root]
        raise ValueError("The ZIP archive did not contain sub.txt and num.txt.")

    if data_path.exists() and data_path.is_dir():
        if (data_path / "sub.txt").exists() and (data_path / "num.txt").exists():
            return [data_path]

        zip_files = sorted(data_path.glob("*.zip"))
        if zip_files:
            resolved: List[Path] = []
            for archive_path in zip_files:
                extract_root = CACHE_DIR / "extracted" / archive_path.stem
                if extract_root.exists():
                    shutil.rmtree(extract_root)
                extract_root.mkdir(parents=True, exist_ok=True)
                with zipfile.ZipFile(archive_path) as archive:
                    archive.extractall(extract_root)
                if (extract_root / "sub.txt").exists() and (extract_root / "num.txt").exists():
                    resolved.append(extract_root)
                else:
                    raise ValueError(f"{archive_path.name} did not contain both sub.txt and num.txt.")
            return resolved

        raise ValueError("The selected directory must contain sub.txt and num.txt, or ZIP archive(s) containing them.")

    raise ValueError("The data directory does not exist or is not a folder.")


def get_db_connection() -> duckdb.DuckDBPyConnection:
    ensure_cache_dir()
    return duckdb.connect(str(DB_PATH))


def list_available_data_files(data_dir: str) -> List[str]:
    data_path = Path(data_dir).expanduser()
    if not data_path.exists() or not data_path.is_dir():
        return []
    return [p.name for p in sorted(data_path.iterdir()) if p.is_file() and p.suffix.lower() in {".zip", ".txt"}]


def clear_cached_data() -> List[str]:
    """Drop all cached tables (fundamentals, market prices, watchlist, logs)
    and remove the DuckDB file.  Returns a list describing what was cleared.

    This is a destructive operation — after calling it, you must re-ingest
    all quarterly data before the screen works again.
    """
    cleared: List[str] = []

    # Drop tables from the DuckDB file (safer than deleting the file while
    # another connection might have it open).
    con = get_db_connection()
    try:
        tables = con.execute(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema = 'main'"
        ).fetchall()
        for (tbl,) in tables:
            con.execute(f'DROP TABLE IF EXISTS "{tbl}"')
            cleared.append(tbl)
        con.commit()
    finally:
        con.close()

    # Delete the DuckDB file itself so the next ingest starts clean.
    if DB_PATH.exists():
        DB_PATH.unlink()
        cleared.append(f"Deleted {DB_PATH.name}")

    # Clear cached JSON files (ticker map, exchange map, market data).
    from core.paths import MARKET_CACHE_PATH, EXCHANGE_CACHE_PATH, TICKER_CACHE_PATH
    for cache_path in [MARKET_CACHE_PATH, EXCHANGE_CACHE_PATH, TICKER_CACHE_PATH]:
        if cache_path.exists():
            cache_path.unlink()
            cleared.append(f"Deleted {cache_path.name}")

    return cleared