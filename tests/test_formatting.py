"""Tests for core/formatting.py."""

from core.formatting import format_display_value, GROUP_MAP, GROUP_ORDER


class TestCurrencyScaling:
    def test_billions(self):
        assert format_display_value("Revenues", 1_500_000_000) == "$1.50B"

    def test_millions(self):
        assert format_display_value("Revenues", 1_500_000) == "$1.50M"

    def test_thousands(self):
        assert format_display_value("Revenues", 1_500) == "$1.5K"

    def test_hundreds(self):
        assert format_display_value("Revenues", 150) == "$150.00"

    def test_negative(self):
        result = format_display_value("NetIncomeLoss", -5000000)
        assert result == "-$5.00M"


class TestOtherFormats:
    def test_percent(self):
        assert format_display_value("ROIC", 0.1525) == "15.25%"

    def test_multiple(self):
        assert format_display_value("CurrentRatio", 1.524) == "1.52x"

    def test_date(self):
        assert format_display_value("period", 20231231) == "Dec 31, 2023"

    def test_year(self):
        assert format_display_value("fy", 2023.0) == "2023"

    def test_infinity(self):
        result = format_display_value("DebtToEquity", float("inf"))
        assert result == "∞"

    def test_interest_coverage_infinity(self):
        """InterestCoverage = inf should show '(no debt)' specifically."""
        result = format_display_value("InterestCoverage", float("inf"))
        assert result == "∞ (no debt)"
        assert "no debt" in result

    def test_none_returns_emdash(self):
        assert format_display_value("Revenues", None) == "—"

    def test_nan_returns_emdash(self):
        assert format_display_value("ROIC", float("nan")) == "—"


class TestGroupMap:
    def test_cfo_to_ni_is_earnings_quality(self):
        """CFO_to_NI maps to Earnings Quality, NOT Profitability.
        Regression test for a substring-matching bug where 'cfo' matched
        a Profitability keyword first."""
        assert GROUP_MAP["CFO_to_NI"] == "Earnings Quality"

    def test_roic_is_profitability(self):
        assert GROUP_MAP["ROIC"] == "Profitability"

    def test_debt_to_equity_is_leverage(self):
        assert GROUP_MAP["DebtToEquity"] == "Leverage & Solvency"

    def test_group_order_has_all_categories(self):
        assert "Profitability" in GROUP_ORDER
        assert "Leverage & Solvency" in GROUP_ORDER
        assert "Earnings Quality" in GROUP_ORDER
        assert "Market Data" in GROUP_ORDER
