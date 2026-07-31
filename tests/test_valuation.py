"""Tests for core/valuation.py -- Graham Number, DCF, margin of safety."""

import math

import pandas as pd
import pytest

from core.fundamentals.valuation import (
    compute_graham_number,
    compute_dcf_intrinsic_value_per_share,
    compute_margin_of_safety,
)


# ---------------------------------------------------------------------------
# Graham Number
# ---------------------------------------------------------------------------

class TestGrahamNumber:
    def test_exact_value(self):
        """Graham Number = sqrt(22.5 * EPS * BVPS)."""
        result = compute_graham_number(eps=5, book_value_per_share=40)
        expected = math.sqrt(22.5 * 5 * 40)  # 67.082...
        assert result == pytest.approx(expected, rel=1e-6)

    def test_negative_eps_returns_none(self):
        assert compute_graham_number(eps=-1, book_value_per_share=40) is None

    def test_negative_book_value_returns_none(self):
        assert compute_graham_number(eps=5, book_value_per_share=-10) is None

    def test_none_eps_returns_none(self):
        assert compute_graham_number(eps=None, book_value_per_share=40) is None

    def test_none_book_value_returns_none(self):
        assert compute_graham_number(eps=5, book_value_per_share=None) is None

    def test_nan_eps_returns_none(self):
        assert compute_graham_number(eps=float("nan"), book_value_per_share=40) is None

    def test_nan_book_value_returns_none(self):
        assert compute_graham_number(eps=5, book_value_per_share=float("nan")) is None

    def test_pd_na_returns_none(self):
        """pd.NA should be handled like None."""
        result = compute_graham_number(eps=pd.NA, book_value_per_share=40)
        assert result is None


# ---------------------------------------------------------------------------
# DCF Intrinsic Value
# ---------------------------------------------------------------------------

class TestDCF:
    def test_exact_value(self):
        """fcf=100, shares=1000, growth=5%, discount=10%, terminal=2.5%, 5yr.
        Hand-derived: ~1.5189."""
        result = compute_dcf_intrinsic_value_per_share(
            fcf=100, shares_outstanding=1000,
            growth_rate=0.05, discount_rate=0.10,
            terminal_growth_rate=0.025, projection_years=5,
        )
        assert result == pytest.approx(1.5189, rel=1e-4)

    def test_discount_rate_le_terminal_rate_returns_none(self):
        """discount_rate <= terminal_growth_rate is nonsensical."""
        result = compute_dcf_intrinsic_value_per_share(
            fcf=100, shares_outstanding=1000,
            growth_rate=0.05, discount_rate=0.02,
            terminal_growth_rate=0.025, projection_years=5,
        )
        assert result is None

    def test_zero_fcf_returns_none(self):
        result = compute_dcf_intrinsic_value_per_share(
            fcf=0, shares_outstanding=1000,
        )
        assert result is None

    def test_negative_fcf_returns_none(self):
        result = compute_dcf_intrinsic_value_per_share(
            fcf=-50, shares_outstanding=1000,
        )
        assert result is None

    def test_nan_shares_returns_none(self):
        result = compute_dcf_intrinsic_value_per_share(
            fcf=100, shares_outstanding=float("nan"),
        )
        assert result is None

    def test_none_fcf_returns_none(self):
        result = compute_dcf_intrinsic_value_per_share(
            fcf=None, shares_outstanding=1000,
        )
        assert result is None


# ---------------------------------------------------------------------------
# Margin of Safety
# ---------------------------------------------------------------------------

class TestMarginOfSafety:
    def test_adds_columns_when_price_present(self):
        """When Price is present, both margin columns are added."""
        df = pd.DataFrame([{
            "Price": 50.0,
            "CommonStockSharesOutstanding": 1000.0,
            "NetIncomeLoss": 10000.0,
            "StockholdersEquity": 50000.0,
            "FCF": 5000.0,
        }])
        result = compute_margin_of_safety(df)
        assert "MarginOfSafetyGraham" in result.columns
        assert "MarginOfSafetyDCF" in result.columns

    def test_no_price_column_returns_unchanged(self):
        """When Price is absent, DataFrame is returned unchanged."""
        df = pd.DataFrame([{
            "CommonStockSharesOutstanding": 1000.0,
            "NetIncomeLoss": 10000.0,
            "StockholdersEquity": 50000.0,
            "FCF": 5000.0,
        }])
        result = compute_margin_of_safety(df)
        # No new columns, same data
        assert "MarginOfSafetyGraham" not in result.columns
        assert "MarginOfSafetyDCF" not in result.columns
        assert len(result.columns) == len(df.columns)
