"""Tests for core/exchange_filter.py."""

import json
import time
from unittest.mock import patch, MagicMock

import pandas as pd
import pytest

from core.fundamentals.exchange_filter import (
    is_major_exchange,
    filter_to_major_exchanges,
    load_exchange_map,
    get_allowed_cik_set,
)


# ---------------------------------------------------------------------------
# Cache freshness
# ---------------------------------------------------------------------------

class TestCacheFreshness:
    @patch("core.fundamentals.exchange_filter.get_cik_exchange_map")
    @patch("core.fundamentals.exchange_filter.EXCHANGE_CACHE_PATH")
    def test_cache_hit_avoids_refetch(self, mock_path, mock_get_map):
        """Calling load_exchange_map twice within cache TTL only fetches once."""
        # Must have >= MIN_EXCHANGE_ENTRIES (100) entries to pass cache validation
        n_entries = 150
        large_map = {str(i): "Nasdaq" for i in range(n_entries)}
        mock_get_map.return_value = large_map

        # Simulate: cache file exists and is fresh
        mock_path.exists.return_value = True
        mock_stat = MagicMock()
        mock_stat.st_mtime = time.time() - 60  # 60 seconds old (fresh)
        mock_path.stat.return_value = mock_stat
        # Return valid cached data
        mock_path.read_text.return_value = json.dumps(large_map)

        result1, err1 = load_exchange_map()
        result2, err2 = load_exchange_map()

        # The underlying fetch should only be called once (or zero times
        # if the cache hit short-circuits first -- in either case the
        # cached data should be returned both times without a second
        # network call).
        assert mock_get_map.call_count <= 1
        assert result1 == large_map
        assert err1 is None
        assert result2 == result1


# ---------------------------------------------------------------------------
# Graceful degradation
# ---------------------------------------------------------------------------

class TestGracefulDegradation:
    def test_filter_returns_unfiltered_on_fetch_failure(self):
        """When the exchange map fetch fails, filter_to_major_exchanges
        returns the input DataFrame UNFILTERED with a warning."""
        df = pd.DataFrame({
            "cik": ["320193", "999999"],
            "name": ["Apple", "Unknown"],
        })
        with patch("core.fundamentals.exchange_filter.load_exchange_map") as mock_load:
            mock_load.return_value = ({}, "Simulated network failure")
            result, warning = filter_to_major_exchanges(df)
            # DataFrame should be unchanged (both rows present)
            assert len(result) == 2
            assert warning is not None
            assert "Simulated network failure" in warning


# ---------------------------------------------------------------------------
# Substring matching
# ---------------------------------------------------------------------------

class TestSubstringMatching:
    def test_nasdaq_global_select(self):
        assert is_major_exchange("NASDAQ Global Select") is True

    def test_new_york_stock_exchange(self):
        assert is_major_exchange("New York Stock Exchange") is True

    def test_nyse_arca(self):
        assert is_major_exchange("NYSE Arca") is True

    def test_otc_is_not_major(self):
        assert is_major_exchange("OTC") is False

    def test_none_is_not_major(self):
        assert is_major_exchange(None) is False

    def test_empty_string_is_not_major(self):
        assert is_major_exchange("") is False


class TestGetAllowedCikSet:
    def test_filters_to_major_exchanges_only(self):
        exchange_map = {
            "320193": "Nasdaq",
            "789019": "NYSE",
            "999999": "OTC",
            "111111": "",  # empty exchange
        }
        allowed = get_allowed_cik_set(exchange_map)
        assert "320193" in allowed
        assert "789019" in allowed
        assert "999999" not in allowed
        assert "111111" not in allowed
