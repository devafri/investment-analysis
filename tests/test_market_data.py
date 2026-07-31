"""Tests for providers/market_data.py."""

from unittest.mock import patch, MagicMock

import pandas as pd
import pytest

from providers.market_data import _apply_price_result, coerce_float


class TestApplyPriceResult:
    def test_price_and_market_cap_stored(self):
        """Price and derived columns are written into the DataFrame."""
        df = pd.DataFrame({
            "cik": ["12345"],
            "Price": [None], "MarketCap": [None],
            "EnterpriseValue": [None], "EarningsYield": [None],
            "PE": [None], "PB": [None], "EVToEBIT": [None], "PFCF": [None],
            "TotalDebt": [200], "CashAndCashEquivalents": [50],
            "OperatingIncomeLoss": [150], "NetIncomeLoss": [100],
            "StockholdersEquity": [500], "FCF": [120],
            "CommonStockSharesOutstanding": [10_000_000],
        })
        row = df.iloc[0]
        prices = {"price": 50.0, "market_cap": 500_000_000}
        _apply_price_result(df, 0, prices, row)
        assert df.at[0, "Price"] == 50.0
        assert df.at[0, "MarketCap"] == 500_000_000

    def test_enterprise_value_computed(self):
        """EV = MarketCap + TotalDebt - Cash."""
        df = pd.DataFrame({
            "cik": ["12345"],
            "Price": [50.0], "MarketCap": [500_000_000],
            "EnterpriseValue": [None], "EarningsYield": [None],
            "PE": [None], "PB": [None], "EVToEBIT": [None], "PFCF": [None],
            "TotalDebt": [200], "CashAndCashEquivalents": [50],
            "OperatingIncomeLoss": [150], "NetIncomeLoss": [100],
            "StockholdersEquity": [500], "FCF": [120],
            "CommonStockSharesOutstanding": [10_000_000],
        })
        row = df.iloc[0]
        prices = {"price": 50.0, "market_cap": 500_000_000}
        _apply_price_result(df, 0, prices, row)
        # EV = 500M + 200 - 50 = 650M
        assert df.at[0, "EnterpriseValue"] == pytest.approx(500_000_000 + 200 - 50)

    def test_multiples_computed(self):
        """P/E, P/B, EV/EBIT, P/FCF should be populated."""
        df = pd.DataFrame({
            "cik": ["12345"],
            "Price": [None], "MarketCap": [None],
            "EnterpriseValue": [None], "EarningsYield": [None],
            "PE": [None], "PB": [None], "EVToEBIT": [None], "PFCF": [None],
            "TotalDebt": [0], "CashAndCashEquivalents": [0],
            "OperatingIncomeLoss": [100], "NetIncomeLoss": [100],
            "StockholdersEquity": [500], "FCF": [80],
            "CommonStockSharesOutstanding": [10_000_000],
        })
        row = df.iloc[0]
        prices = {"price": 50.0, "market_cap": 500_000_000}
        _apply_price_result(df, 0, prices, row)
        assert df.at[0, "PE"] is not None  # 50 / (100/10) = 5
        assert df.at[0, "PB"] is not None  # 50 / (500/10) = 1
        assert df.at[0, "EVToEBIT"] is not None  # 500M/100 = 5
        assert df.at[0, "PFCF"] is not None  # 500M/80 = 6.25


class TestSchwabBranching:
    @patch("providers.market_data._fetch_via_schwab")
    @patch("providers.market_data.get_cik_ticker_map")
    def test_schwab_path_used(self, mock_cik_map, mock_schwab):
        """join_market_data always uses the Schwab branch."""
        from providers.market_data import join_market_data

        mock_cik_map.return_value = {"12345": "TEST"}
        mock_schwab.return_value = []

        df = pd.DataFrame({
            "CIK": ["12345"],
            "ROIC": [0.2],
            "TotalDebt": [100],
            "CashAndCashEquivalents": [50],
            "OperatingIncomeLoss": [200],
        })

        result, errors = join_market_data(df)
        mock_schwab.assert_called_once()
