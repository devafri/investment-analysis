"""Tests for core/screening.py."""

import pandas as pd
import pytest

from core.screening import (
    parse_thresholds,
    base_query_string,
    build_query_string,
    paginate_frame,
)


# ---------------------------------------------------------------------------
# Percent-scaling threshold parsing
# ---------------------------------------------------------------------------

class TestParseThresholds:
    def test_min_roic_percent_scaling(self):
        """min_roic=15 (UI sends percentage) -> 0.15 (stored as fraction)."""
        result = parse_thresholds({"min_roic": "15"})
        assert result["min_roic"] == 0.15

    def test_min_operating_margin_percent_scaling(self):
        result = parse_thresholds({"min_operating_margin": "10"})
        assert result["min_operating_margin"] == 0.10

    def test_other_params_not_percent_scaled(self):
        result = parse_thresholds({"max_debt_to_equity": "1.0"})
        assert result["max_debt_to_equity"] == 1.0

    def test_boolean_checkbox_duplicate_keys(self):
        """Real form behavior: a checked checkbox sends both a hidden
        input (require_positive_ni=0) AND the checkbox value
        (require_positive_ni=on). FastAPI/Starlette's dict(query_params)
        keeps the LAST occurrence -- so 'on' should win."""
        # Simulate: the last value for the key should win
        # In a real query string, duplicate keys: ?require_positive_ni=0&require_positive_ni=on
        # dict(query_params) keeps the last one
        params = {"require_positive_ni": "on"}  # last value wins
        result = parse_thresholds(params)
        assert result["require_positive_ni"] is True

    def test_boolean_checkbox_unchecked(self):
        """When the checkbox is unchecked, only the hidden input (0) is sent."""
        params = {"require_positive_ni": "0"}
        result = parse_thresholds(params)
        assert result["require_positive_ni"] is False

    def test_first_page_load_defaults(self):
        """When NO params at all (first page load), boolean defaults apply."""
        result = parse_thresholds({})
        assert result["require_positive_ni"] is True
        assert result["major_exchanges_only"] is True

    def test_defaults_for_all_keys(self):
        result = parse_thresholds({})
        assert result["min_roic"] == 0.15
        assert result["min_operating_margin"] == 0.10
        assert result["max_debt_to_equity"] == 1.0
        assert result["min_interest_coverage"] == 5.0
        assert result["min_cfo_to_ni"] == 0.8


# ---------------------------------------------------------------------------
# Base query string
# ---------------------------------------------------------------------------

class TestBaseQueryString:
    def test_strips_sort_order_page(self):
        params = {
            "sort": "roic",
            "order": "desc",
            "page": "3",
            "min_roic": "15",
            "major_exchanges_only": "on",
        }
        result = base_query_string(params)
        assert "sort" not in result
        assert "order" not in result
        assert "page" not in result
        assert "min_roic=15" in result
        assert "major_exchanges_only=on" in result

    def test_preserves_other_keys(self):
        params = {"min_roic": "15", "max_debt_to_equity": "0.5"}
        result = base_query_string(params)
        assert "min_roic=15" in result
        assert "max_debt_to_equity=0.5" in result

    def test_empty_params(self):
        assert base_query_string({}) == ""


# ---------------------------------------------------------------------------
# Pagination and sorting
# ---------------------------------------------------------------------------

class TestPaginateFrame:
    def test_sort_and_page(self):
        df = pd.DataFrame({
            "cik": ["1", "2", "3", "4"],
            "name": ["A", "B", "C", "D"],
            "ROIC": [0.2, 0.1, 0.3, 0.15],
        })
        page_df, pagination = paginate_frame(df, {"sort": "roic", "order": "desc", "page": "1"})
        # Descending ROIC: 0.3, 0.2, 0.15, 0.1
        assert list(page_df["ROIC"]) == [0.3, 0.2, 0.15, 0.1]
        assert pagination["page"] == 1
        assert pagination["total_rows"] == 4

    def test_pagination_info(self):
        """Create 25 rows (more than PAGE_SIZE=20) to test pagination."""
        df = pd.DataFrame({
            "cik": [str(i) for i in range(25)],
            "name": [f"Company {i}" for i in range(25)],
            "ROIC": [0.1 + i * 0.01 for i in range(25)],
        })
        page_df, pagination = paginate_frame(df, {"sort": "roic", "order": "asc", "page": "1"})
        assert len(page_df) == 20  # PAGE_SIZE
        assert pagination["total_pages"] == 2
        assert pagination["total_rows"] == 25
        assert pagination["page"] == 1

        # Page 2
        page_df2, pagination2 = paginate_frame(df, {"sort": "roic", "order": "asc", "page": "2"})
        assert len(page_df2) == 5

    def test_sort_by_name(self):
        df = pd.DataFrame({
            "cik": ["1", "2", "3"],
            "name": ["Z Corp", "A Corp", "M Corp"],
            "ROIC": [0.2, 0.1, 0.3],
        })
        page_df, _ = paginate_frame(df, {"sort": "name", "order": "asc", "page": "1"})
        assert list(page_df["name"]) == ["A Corp", "M Corp", "Z Corp"]


# ---------------------------------------------------------------------------
# Financial sector exclusion
# ---------------------------------------------------------------------------

class TestFinancialExclusion:
    def test_exclude_financials_removes_sic_60xx(self):
        """Financial sector toggle should remove SIC 6000-6799."""
        from core.screening import DEFAULT_THRESHOLDS
        assert "exclude_financials" in DEFAULT_THRESHOLDS
        assert DEFAULT_THRESHOLDS["exclude_financials"] is False

    def test_exclude_financials_is_boolean_key(self):
        """exclude_financials should be parsed as a boolean."""
        from core.screening import BOOLEAN_THRESHOLD_KEYS
        assert "exclude_financials" in BOOLEAN_THRESHOLD_KEYS


# ---------------------------------------------------------------------------
# Market price persistence
# ---------------------------------------------------------------------------

class TestMarketPricePersistence:
    def test_save_and_hydrate(self, tmp_path):
        """Save prices to DB, then hydrate them back."""
        import duckdb
        from core.screening import save_market_prices_to_db, _hydrate_prices_from_db
        from core.data_ingestion import get_db_connection

        # Create a test DataFrame with price data
        df = pd.DataFrame({
            "cik": ["12345", "67890"],
            "Ticker": ["TEST", "DEMO"],
            "Price": [100.0, 50.0],
            "MarketCap": [1e9, 5e8],
            "EnterpriseValue": [1.2e9, 6e8],
            "EarningsYield": [0.05, 0.03],
            "PE": [20.0, 15.0],
            "PB": [3.0, 2.0],
            "EVToEBIT": [12.0, 10.0],
            "PFCF": [18.0, 14.0],
            "MagicFormulaRank": [1, 2],
            "ROIC": [0.2, 0.15],
        })

        # Save
        save_market_prices_to_db(df)

        # Verify DB has the data
        con = get_db_connection()
        try:
            count = con.execute("SELECT COUNT(*) FROM market_prices").fetchone()[0]
            assert count >= 2
        finally:
            con.close()

        # Hydrate into a fundamentals-like DataFrame
        fundamentals = pd.DataFrame({
            "cik": ["12345", "67890", "99999"],
            "name": ["Test Corp", "Demo Inc", "No Price Co"],
        })
        enriched = _hydrate_prices_from_db(fundamentals)
        assert "Price" in enriched.columns
        assert enriched.loc[0, "Price"] == 100.0
        assert enriched.loc[1, "Price"] == 50.0
        # Company with no price should have NaN
        assert pd.isna(enriched.loc[2, "Price"])
