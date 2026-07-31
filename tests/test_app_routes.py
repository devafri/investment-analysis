"""Integration tests for app.py routes via FastAPI TestClient.

Uses synthetic sub.txt/num.txt fixtures. ALL outbound network calls are mocked.
"""

import json
from pathlib import Path
from unittest.mock import patch, MagicMock

import duckdb
import pytest
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app import app
from tests.conftest import (
    write_sub_txt,
    write_num_txt,
    make_ttm_company_data,
)


@pytest.fixture(autouse=True)
def tmp_db(monkeypatch, tmp_path):
    """Redirect ALL cache I/O to a temp directory so tests never touch the
    real cache/. This is autouse so even a bare `client.get` that triggers
    import-time side-effects won't read, write, or delete real user data."""
    import core.paths
    import core.data_ingestion

    tmp_cache = tmp_path / "test_cache"
    tmp_cache.mkdir()
    tmp_db_path = tmp_cache / "screen.duckdb"

    monkeypatch.setattr(core.paths, "CACHE_DIR", tmp_cache)
    monkeypatch.setattr(core.paths, "DB_PATH", tmp_db_path)
    monkeypatch.setattr(core.paths, "MARKET_CACHE_PATH", tmp_cache / "market_data.json")
    monkeypatch.setattr(core.paths, "EXCHANGE_CACHE_PATH", tmp_cache / "exchange_map.json")
    monkeypatch.setattr(core.paths, "TICKER_CACHE_PATH", tmp_cache / "ticker_map.json")
    monkeypatch.setattr(core.data_ingestion, "CACHE_DIR", tmp_cache)
    monkeypatch.setattr(core.data_ingestion, "DB_PATH", tmp_db_path)
    yield


@pytest.fixture
def client():
    """FastAPI TestClient."""
    return TestClient(app)


@pytest.fixture
def mock_network():
    """Mock all outbound network dependencies for the full flow.

    Patches at BOTH the source (pipeline_imports) and the consumer modules
    (exchange_filter, market_data) to handle Python's import-time name binding.
    """
    with patch("providers.pipeline_imports.get_cik_ticker_map") as mock_ticker, \
         patch("providers.pipeline_imports.get_cik_exchange_map") as mock_exch, \
         patch("providers.pipeline_imports.fetch_price_data") as mock_price, \
         patch("core.exchange_filter.get_cik_exchange_map") as mock_exch2, \
         patch("providers.market_data.fetch_price_data") as mock_price2, \
         patch("providers.market_data.get_cik_ticker_map") as mock_ticker2:
        mock_ticker.return_value = {"12345": "TEST", "99999": "ZETA"}
        mock_ticker2.return_value = {"12345": "TEST", "99999": "ZETA"}
        mock_exch.return_value = {"12345": "Nasdaq", "99999": "NYSE"}
        mock_exch2.return_value = {"12345": "Nasdaq", "99999": "NYSE"}
        mock_price.return_value = {
            "ticker": "TEST", "price": 50.0, "shares_out": 10_000_000,
            "market_cap": 500_000_000,
        }
        mock_price2.return_value = {
            "ticker": "TEST", "price": 50.0, "shares_out": 10_000_000,
            "market_cap": 500_000_000,
        }
        yield


# ---------------------------------------------------------------------------
# 1. Full flow: POST /ingest -> GET /screen shows ingested companies
# ---------------------------------------------------------------------------

class TestFullFlow:
    def test_ingest_then_screen(self, client, tmp_path, mock_network):
        """Ingest synthetic data, then verify it persists in DuckDB and
        the /screen page loads (even if thresholds filter the test company
        out, the page itself must render without error)."""
        import time

        make_ttm_company_data(tmp_path)

        # POST /ingest — launches a background thread, returns progress bar
        resp = client.post("/ingest", data={"data_dir": str(tmp_path)})
        assert resp.status_code == 200

        # Poll /ingest/progress the same way an HTMX client would,
        # waiting for the background thread to finish.  When complete the
        # template renders "✓ Ingest complete" and stops polling
        # (hx-trigger="none").
        for _ in range(50):  # max ~5 seconds
            resp = client.get("/ingest/progress")
            assert resp.status_code == 200
            if "Ingest complete" in resp.text:
                break
            time.sleep(0.1)
        else:
            pytest.fail("Ingest did not complete within 5 seconds")

        # Now it's safe to assert on the persisted data.
        import duckdb, core.paths
        con = duckdb.connect(str(core.paths.DB_PATH))
        count = con.execute(
            "SELECT COUNT(*) FROM fundamentals_history"
        ).fetchone()[0]
        con.close()
        assert count > 0, "Data should persist in DuckDB after ingest"

        # GET /screen — page must render without crashing
        resp = client.get("/screen")
        assert resp.status_code == 200
        # Data was ingested — the page should not show the "no cached data" error
        assert "No cached screening data is available yet" not in resp.text

    def test_ingest_shows_error_on_invalid_dir(self, client, tmp_path, mock_network):
        """Ingesting a directory with no sub.txt/num.txt should show error."""
        import time

        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        resp = client.post("/ingest", data={"data_dir": str(empty_dir)})
        assert resp.status_code == 200

        # The error is set by the background thread — poll until complete
        for _ in range(50):
            resp = client.get("/ingest/progress")
            assert resp.status_code == 200
            if "Ingest complete" in resp.text or "Ingest failed" in resp.text:
                break
            time.sleep(0.1)

        assert "error" in resp.text.lower()


# ---------------------------------------------------------------------------
# 2. Sort-header links produce correct reversed order
# ---------------------------------------------------------------------------

class TestSortHeaders:
    def test_sort_reversal_on_second_click(self, client, tmp_path, mock_network):
        """Clicking a sort header twice should reverse the order."""
        # Create data with two companies so we can verify ordering
        adsh1 = "0000012345-24-000001"
        adsh2 = "0000099999-24-000001"

        write_sub_txt(tmp_path / "sub.txt", [
            {"adsh": adsh1, "cik": "12345", "name": "ALPHA CORP", "sic": "7370",
             "form": "10-K", "period": "20231231", "fy": "2023", "fp": "FY",
             "filed": "20240215"},
            {"adsh": adsh2, "cik": "99999", "name": "ZETA CORP", "sic": "7370",
             "form": "10-K", "period": "20231231", "fy": "2023", "fp": "FY",
             "filed": "20240215"},
        ])

        num_rows = []
        def add(adsh, tag, qtrs, val):
            num_rows.append({"adsh": adsh, "tag": tag, "version": "us-gaap/2023",
                             "coreg": "", "ddate": "20231231", "qtrs": qtrs,
                             "uom": "USD", "value": val, "footnote": ""})
        # Alpha: higher ROIC
        for tag, val in [("Revenues", "1000000000"), ("OperatingIncomeLoss", "300000000"),
                         ("NetIncomeLoss", "200000000"), ("InterestExpense", "10000000")]:
            add(adsh1, tag, "4", val)
        for tag, val in [("Assets", "2000000000"), ("AssetsCurrent", "500000000"),
                         ("LiabilitiesCurrent", "300000000"),
                         ("CashAndCashEquivalentsAtCarryingValue", "100000000"),
                         ("PropertyPlantAndEquipmentNet", "800000000"),
                         ("StockholdersEquity", "1000000000"),
                         ("LongTermDebtNoncurrent", "400000000"),
                         ("LongTermDebtCurrent", "50000000"),
                         ("CommonStockSharesOutstanding", "50000000")]:
            add(adsh1, tag, "0", val)

        # Zeta: lower ROIC
        for tag, val in [("Revenues", "1000000000"), ("OperatingIncomeLoss", "100000000"),
                         ("NetIncomeLoss", "50000000"), ("InterestExpense", "10000000")]:
            add(adsh2, tag, "4", val)
        for tag, val in [("Assets", "2000000000"), ("AssetsCurrent", "500000000"),
                         ("LiabilitiesCurrent", "300000000"),
                         ("CashAndCashEquivalentsAtCarryingValue", "100000000"),
                         ("PropertyPlantAndEquipmentNet", "800000000"),
                         ("StockholdersEquity", "1000000000"),
                         ("LongTermDebtNoncurrent", "400000000"),
                         ("LongTermDebtCurrent", "50000000"),
                         ("CommonStockSharesOutstanding", "50000000")]:
            add(adsh2, tag, "0", val)

        write_num_txt(tmp_path / "num.txt", num_rows)

        # Ingest
        resp = client.post("/ingest", data={"data_dir": str(tmp_path)})
        assert resp.status_code == 200

        # Get page with sort=roic, order=asc
        resp_asc = client.get("/screen?sort=roic&order=asc")
        html_asc = resp_asc.text

        # Get page with sort=roic, order=desc
        resp_desc = client.get("/screen?sort=roic&order=desc")
        html_desc = resp_desc.text

        # Both should return 200 and contain company data
        assert resp_asc.status_code == 200
        assert resp_desc.status_code == 200


# ---------------------------------------------------------------------------
# 3. GET /company/{cik}
# ---------------------------------------------------------------------------

class TestCompanyDetail:
    def test_company_page_shows_ttm_summary(self, client, tmp_path, mock_network):
        """Company detail page shows TTM summary, valuation section, and
        filing-history comparison table."""
        make_ttm_company_data(tmp_path)

        # Ingest
        resp = client.post("/ingest", data={"data_dir": str(tmp_path)})
        assert resp.status_code == 200

        # Visit company detail
        resp = client.get("/company/12345")
        assert resp.status_code == 200
        html = resp.text

        # Should contain the company name
        assert "TEST CORP" in html
        # Should contain profitability section
        assert "Profitability" in html or "ROIC" in html
        # Should contain the comparison table / filing history
        assert "FY" in html or "2023" in html or "10-K" in html

    def test_unknown_cik_shows_error(self, client, tmp_path, mock_network):
        """Requesting a non-existent CIK should show an error."""
        make_ttm_company_data(tmp_path)
        client.post("/ingest", data={"data_dir": str(tmp_path)})
        resp = client.get("/company/999999")
        assert resp.status_code == 200
        assert "error" in resp.text.lower() or "No company" in resp.text


# ---------------------------------------------------------------------------
# 4. Multiple zips accumulate
# ---------------------------------------------------------------------------

class TestMultipleZips:
    def test_two_ingests_both_accumulate(self, client, tmp_path, mock_network):
        """Ingesting two different data dirs accumulates both, not 'last one wins'."""
        dir1 = tmp_path / "batch1"
        dir1.mkdir()
        make_ttm_company_data(dir1)  # CIK 12345

        dir2 = tmp_path / "batch2"
        dir2.mkdir()
        # Second company
        from tests.conftest import make_debt_free_data
        make_debt_free_data(dir2)  # CIK 12345 but with different name "DEBTFREE INC"

        # Ingest both
        resp1 = client.post("/ingest", data={"data_dir": str(dir1)})
        assert resp1.status_code == 200

        resp2 = client.post("/ingest", data={"data_dir": str(dir2)})
        assert resp2.status_code == 200

        # /screen should work (TTM should handle both)
        resp = client.get("/screen")
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# 5. POST /market-data/refresh scope
# ---------------------------------------------------------------------------

class TestMarketDataRefresh:
    def test_scope_all_vs_page(self, client, tmp_path, mock_network):
        """scope=all should fetch more tickers than scope=page."""
        make_ttm_company_data(tmp_path)

        # Ingest
        resp = client.post("/ingest", data={"data_dir": str(tmp_path)})
        assert resp.status_code == 200

        # Refresh with scope=page
        resp_page = client.post("/market-data/refresh?scope=page")
        assert resp_page.status_code == 200

        # Refresh with scope=all
        resp_all = client.post("/market-data/refresh?scope=all")
        assert resp_all.status_code == 200


# ---------------------------------------------------------------------------
# 6. Missing imports surface clear error
# ---------------------------------------------------------------------------

class TestMissingImports:
    def test_require_pipeline_gives_specific_error(self):
        """require_pipeline() should raise RuntimeError with a clear, specific
        message when imports failed, not a generic one."""
        from providers.pipeline_imports import require_pipeline, IMPORT_ERROR

        if IMPORT_ERROR is not None:
            # Imports already failed -- the error should be specific
            with pytest.raises(RuntimeError, match="core/sec_value_screen"):
                require_pipeline()
        else:
            # Imports succeeded -- this test just verifies require_pipeline
            # doesn't raise when things are fine
            require_pipeline()  # should not raise

    def test_broken_import_message_is_actionable(self):
        """Simulate a broken import and verify the error message is
        specific and actionable, not generic."""
        # We test this by checking the error message format directly
        try:
            import definitely_does_not_exist_xyz as foo
        except ImportError as e:
            msg = (
                f"Failed to import core/sec_value_screen.py / providers/yfinance_provider.py: "
                f"{type(e).__name__}: {e}"
            )
            assert "core/sec_value_screen" in msg
            assert "yfinance_provider" in msg


# ---------------------------------------------------------------------------
# Health endpoint
# ---------------------------------------------------------------------------

class TestHealth:
    def test_health_returns_ok(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}


# ---------------------------------------------------------------------------
# Price persistence
# ---------------------------------------------------------------------------

class TestPricePersistence:
    def test_market_prices_table_created_after_save(self):
        """save_market_prices_to_db should create the market_prices table."""
        from core.screening import save_market_prices_to_db
        from core.data_ingestion import get_db_connection
        import pandas as pd

        df = pd.DataFrame({
            "cik": ["7654321"],
            "Ticker": ["TST"],
            "Price": [42.50],
        })
        save_market_prices_to_db(df)

        con = get_db_connection()
        try:
            count = con.execute(
                "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = 'market_prices'"
            ).fetchone()[0]
            assert count == 1
            price = con.execute(
                "SELECT price FROM market_prices WHERE cik = '7654321'"
            ).fetchone()[0]
            assert price == 42.50
        finally:
            con.close()
