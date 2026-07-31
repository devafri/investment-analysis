"""Tests for providers/market_data.py."""

import time
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch, MagicMock

import pandas as pd
import pytest

from providers.market_data import (
    _apply_price_result,
    coerce_float,
)


# ---------------------------------------------------------------------------
# Timeout doesn't block indefinitely
# ---------------------------------------------------------------------------

class TestTimeout:
    def test_timeout_does_not_block_indefinitely(self):
        """Regression test: a slow ticker should not block the entire
        join_market_data call beyond the overall_timeout_seconds deadline.

        We test this at the _fetch_via_yfinance level by mocking
        fetch_price_data: one ticker sleeps forever, one returns fast.
        """
        from providers.market_data import _fetch_via_yfinance

        # The function under test imports fetch_price_data from its own module
        # namespace. We must patch it THERE (where it's used), not at the source.
        def slow_fetch(ticker):
            if ticker == "SLOW":
                import time
                time.sleep(60)
                return {"ticker": "SLOW", "price": None}
            return {"ticker": "FAST", "price": 100.0, "shares_out": 1_000_000}

        df = pd.DataFrame({
            "cik": ["1", "2"],
            "Ticker": ["SLOW", "FAST"],
            "Price": [None, None],
            "MarketCap": [None, None],
            "EnterpriseValue": [None, None],
            "EarningsYield": [None, None],
            "MagicFormulaRank": [None, None],
        })
        index_to_ticker = {0: "SLOW", 1: "FAST"}

        import time
        with patch("providers.market_data.fetch_price_data", side_effect=slow_fetch):
            start = time.monotonic()
            errors = _fetch_via_yfinance(df, index_to_ticker, overall_timeout_seconds=20)
            elapsed = time.monotonic() - start

        # Should return in roughly 20s, not 60s
        assert elapsed < 30.0, f"Took {elapsed:.1f}s -- timeout didn't work"

        # The fast ticker should still have its data
        assert df.at[1, "Price"] == 100.0

        # The slow ticker should be reported as an error
        assert any("SLOW" in e for e in errors)


# ---------------------------------------------------------------------------
# Provider branching
# ---------------------------------------------------------------------------

class TestProviderBranching:
    @patch("providers.market_data._fetch_via_schwab")
    @patch("providers.market_data._fetch_via_yfinance")
    @patch("providers.market_data.get_cik_ticker_map")
    def test_schwab_branch_called_when_configured(
        self, mock_cik_map, mock_yf, mock_schwab
    ):
        """With MARKET_DATA_PROVIDER='schwab', the Schwab path is used."""
        from providers.market_data import join_market_data, config

        mock_cik_map.return_value = {"12345": "TEST"}
        mock_schwab.return_value = []

        df = pd.DataFrame({
            "CIK": ["12345"],
            "ROIC": [0.2],
            "TotalDebt": [100],
            "CashAndCashEquivalents": [50],
            "OperatingIncomeLoss": [200],
        })

        with patch.object(config, "MARKET_DATA_PROVIDER", "schwab"):
            result, errors = join_market_data(df)

        mock_schwab.assert_called_once()
        mock_yf.assert_not_called()

    @patch("providers.market_data._fetch_via_yfinance")
    @patch("providers.market_data._fetch_via_schwab")
    @patch("providers.market_data.get_cik_ticker_map")
    def test_yfinance_branch_called_when_configured(
        self, mock_cik_map, mock_schwab, mock_yf
    ):
        """With MARKET_DATA_PROVIDER='yfinance', the yfinance path is used."""
        from providers.market_data import join_market_data, config

        mock_cik_map.return_value = {"12345": "TEST"}
        mock_yf.return_value = []

        df = pd.DataFrame({
            "CIK": ["12345"],
            "ROIC": [0.2],
        })

        with patch.object(config, "MARKET_DATA_PROVIDER", "yfinance"):
            result, errors = join_market_data(df)

        mock_yf.assert_called_once()
        mock_schwab.assert_not_called()

    @patch("providers.market_data._fetch_via_schwab")
    @patch("providers.market_data.get_cik_ticker_map")
    def test_schwab_batch_called_once_for_n_tickers(self, mock_cik_map, mock_schwab):
        """Schwab path calls batch fetch exactly ONCE for N tickers."""
        from providers.market_data import join_market_data, config

        mock_cik_map.return_value = {
            "1": "A", "2": "B", "3": "C", "4": "D", "5": "E",
        }
        mock_schwab.return_value = []

        df = pd.DataFrame({
            "CIK": ["1", "2", "3", "4", "5"],
            "ROIC": [0.2, 0.3, 0.1, 0.25, 0.15],
            "TotalDebt": [0, 0, 0, 0, 0],
            "CashAndCashEquivalents": [0, 0, 0, 0, 0],
            "OperatingIncomeLoss": [100, 200, 150, 300, 250],
        })

        with patch.object(config, "MARKET_DATA_PROVIDER", "schwab"):
            result, errors = join_market_data(df)

        # _fetch_via_schwab should be called once (which internally does one batch call)
        assert mock_schwab.call_count == 1


# ---------------------------------------------------------------------------
# Enterprise Value / Earnings Yield correctness
# ---------------------------------------------------------------------------

class TestEnterpriseValueAndEarningsYield:
    def test_ev_formula(self):
        """EV = MarketCap + TotalDebt - Cash (not just MarketCap)."""
        df = pd.DataFrame({
            "cik": ["1"],
            "Price": [None],
            "MarketCap": [None],
            "EnterpriseValue": [None],
            "EarningsYield": [None],
            "TotalDebt": [500000000],
            "CashAndCashEquivalents": [200000000],
            "OperatingIncomeLoss": [150000000],
        })
        prices = {"ticker": "TEST", "price": 50.0, "shares_out": 20_000_000}
        # MarketCap = 50 * 20M = 1B
        _apply_price_result(df, 0, prices, row=df.iloc[0])

        expected_market_cap = 50.0 * 20_000_000  # 1,000,000,000
        assert df.at[0, "MarketCap"] == expected_market_cap
        # EV = 1B + 500M - 200M = 1.3B
        expected_ev = expected_market_cap + 500_000_000 - 200_000_000
        assert df.at[0, "EnterpriseValue"] == expected_ev
        # EarningsYield = OperatingIncomeLoss / EV = 150M / 1.3B
        expected_ey = 150_000_000 / expected_ev
        assert df.at[0, "EarningsYield"] == pytest.approx(expected_ey)

    def test_ey_uses_operating_income_not_net_income(self):
        """EarningsYield = OperatingIncomeLoss / EV (Greenblatt's definition),
        NOT NetIncomeLoss / MarketCap."""
        df = pd.DataFrame({
            "cik": ["1"],
            "Price": [None],
            "MarketCap": [None],
            "EnterpriseValue": [None],
            "EarningsYield": [None],
            "TotalDebt": [0],
            "CashAndCashEquivalents": [0],
            "OperatingIncomeLoss": [150_000_000],
            "NetIncomeLoss": [100_000_000],  # different from OpInc
        })
        prices = {"ticker": "TEST", "price": 50.0, "shares_out": 20_000_000}
        _apply_price_result(df, 0, prices, row=df.iloc[0])

        # If bug existed: EarningsYield = NetIncomeLoss / MarketCap = 100M/1B = 0.1
        # Correct: EarningsYield = OperatingIncomeLoss / EV = 150M/1B = 0.15
        assert df.at[0, "EarningsYield"] == pytest.approx(0.15)


# ---------------------------------------------------------------------------
# Price is stored, not discarded
# ---------------------------------------------------------------------------

class TestPriceStored:
    def test_price_column_populated(self):
        """After _apply_price_result, the Price column must be populated.
        Regression: a past bug fetched price, used it for market cap,
        then threw it away."""
        df = pd.DataFrame({
            "cik": ["1"],
            "Price": [None],
            "MarketCap": [None],
            "EnterpriseValue": [None],
            "EarningsYield": [None],
            "TotalDebt": [0],
            "CashAndCashEquivalents": [0],
            "OperatingIncomeLoss": [100_000_000],
        })
        prices = {"ticker": "TEST", "price": 75.50, "shares_out": 10_000_000}
        _apply_price_result(df, 0, prices, row=df.iloc[0])
        assert df.at[0, "Price"] == 75.50
        assert df.at[0, "MarketCap"] == 75.50 * 10_000_000


# ---------------------------------------------------------------------------
# Shares-outstanding fallback
# ---------------------------------------------------------------------------

class TestSharesOutstandingFallback:
    def test_fallback_to_sec_shares(self):
        """When provider gives shares_out=None but the SEC row has
        CommonStockSharesOutstanding, MarketCap still gets computed."""
        df = pd.DataFrame({
            "cik": ["1"],
            "Price": [None],
            "MarketCap": [None],
            "EnterpriseValue": [None],
            "EarningsYield": [None],
            "TotalDebt": [0],
            "CashAndCashEquivalents": [0],
            "OperatingIncomeLoss": [100_000_000],
            "CommonStockSharesOutstanding": 50_000_000,
        })
        # Provider has price but no shares_out and no market_cap
        prices = {"ticker": "TEST", "price": 40.0, "shares_out": None, "market_cap": None}
        _apply_price_result(df, 0, prices, row=df.iloc[0])
        # MarketCap = 40 * 50M (SEC figure) = 2B
        assert df.at[0, "MarketCap"] == 2_000_000_000
        assert df.at[0, "Price"] == 40.0
