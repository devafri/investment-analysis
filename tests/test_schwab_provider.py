"""Tests for providers/schwab/market_data.py."""

from unittest.mock import patch, MagicMock

import pytest

from providers.schwab.market_data import (
    _parse_quote_entry,
    _format_error_response,
    fetch_prices_batch,
    SchwabQuoteError,
)


# ---------------------------------------------------------------------------
# Quote parsing against realistic fixtures
# ---------------------------------------------------------------------------

class TestQuoteParsing:
    def test_full_equity(self):
        """AAPL-like response with all fields populated."""
        entry = {
            "quote": {
                "lastPrice": 195.50,
                "mark": 195.45,
            },
            "regular": {
                "regularMarketLastPrice": 195.48,
            },
            "fundamental": {
                "sharesOutstanding": 15_500_000_000,
                "marketCap": 3_030_000_000_000,
                "bookValuePerShare": 4.15,
                "epsTTM": 6.42,
            },
        }
        result = _parse_quote_entry("AAPL", entry)
        assert result["ticker"] == "AAPL"
        assert result["price"] == 195.50
        assert result["shares_out"] == 15_500_000_000
        assert result["market_cap"] == 3_030_000_000_000
        assert result["book_value_per_share"] == 4.15
        assert result["eps_ttm"] == 6.42

    def test_index_no_regular_or_fundamental(self):
        """$SPX-like: no regular or fundamental sections at all.
        Should still extract price from quote section."""
        entry = {
            "quote": {
                "lastPrice": 5200.00,
            },
        }
        result = _parse_quote_entry("$SPX", entry)
        assert result["ticker"] == "$SPX"
        assert result["price"] == 5200.00
        assert result["shares_out"] is None
        assert result["market_cap"] is None

    def test_only_regular_market_last_price(self):
        """Entry with only regular.regularMarketLastPrice (no quote.lastPrice)."""
        entry = {
            "quote": {},
            "regular": {
                "regularMarketLastPrice": 42.50,
            },
            "fundamental": {},
        }
        result = _parse_quote_entry("TICK", entry)
        assert result["ticker"] == "TICK"
        assert result["price"] == 42.50

    def test_missing_price_raises(self):
        """Entry with no price field at all should raise SchwabQuoteError."""
        entry = {"quote": {}, "regular": {}, "fundamental": {}}
        with pytest.raises(SchwabQuoteError, match="Could not find a price field"):
            _parse_quote_entry("BROKEN", entry)


# ---------------------------------------------------------------------------
# Batch fetch: one call for many tickers
# ---------------------------------------------------------------------------

class TestBatchFetch:
    @patch("providers.schwab.market_data.requests.get")
    @patch("providers.schwab.market_data.get_valid_access_token")
    def test_one_http_call_for_many_tickers(self, mock_token, mock_get):
        mock_token.return_value = "test-access-token"
        mock_resp = MagicMock()
        mock_resp.ok = True
        mock_resp.json.return_value = {
            "AAPL": {
                "quote": {"lastPrice": 195.0},
                "regular": {},
                "fundamental": {"sharesOutstanding": 15_500_000_000, "marketCap": 3_000_000_000_000},
            },
            "MSFT": {
                "quote": {"lastPrice": 420.0},
                "regular": {},
                "fundamental": {"sharesOutstanding": 7_430_000_000, "marketCap": 3_120_000_000_000},
            },
            # GOOGL missing from response
        }
        mock_get.return_value = mock_resp

        results = fetch_prices_batch(["AAPL", "MSFT", "GOOGL"])

        # Exactly ONE HTTP call
        assert mock_get.call_count == 1

        # Both present tickers resolved
        assert "AAPL" in results
        assert "MSFT" in results
        assert results["AAPL"]["price"] == 195.0
        assert results["MSFT"]["price"] == 420.0

        # Missing ticker is simply absent, not an exception
        assert "GOOGL" not in results

    @patch("providers.schwab.market_data.requests.get")
    @patch("providers.schwab.market_data.get_valid_access_token")
    def test_empty_ticker_list(self, mock_token, mock_get):
        """Empty ticker list should return empty dict without any HTTP call."""
        results = fetch_prices_batch([])
        assert results == {}
        mock_get.assert_not_called()


# ---------------------------------------------------------------------------
# Structured error parsing
# ---------------------------------------------------------------------------

class TestErrorParsing:
    def test_formatted_error_includes_detail(self):
        """A Schwab-style errors array should produce a message with the detail text."""
        mock_resp = MagicMock()
        mock_resp.status_code = 400
        mock_resp.json.return_value = {
            "errors": [
                {
                    "title": "Bad Request",
                    "detail": "The symbol 'INVALID!!' is not recognized.",
                    "source": {"parameter": "symbols"},
                }
            ]
        }
        msg = _format_error_response(mock_resp)
        assert "400" in msg
        assert "INVALID!!" in msg
        assert "not recognized" in msg

    def test_fallback_to_raw_text(self):
        """Non-JSON response falls back to raw text."""
        mock_resp = MagicMock()
        mock_resp.status_code = 500
        mock_resp.json.side_effect = ValueError("not JSON")
        mock_resp.text = "Internal Server Error"
        msg = _format_error_response(mock_resp)
        assert "500" in msg
        assert "Internal Server Error" in msg
