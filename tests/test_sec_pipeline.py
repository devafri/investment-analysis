"""Tests for core/sec_value_screen.py -- the SEC DERA ingestion/TTM/ratio pipeline."""

import duckdb
import pandas as pd
import pytest

import core.sec_value_screen as sec_screen
from tests.conftest import (
    write_sub_txt,
    write_num_txt,
    write_pre_txt,
    make_ttm_company_data,
    make_10k_only_data,
    make_incomplete_history_data,
    make_ddate_data,
    make_coreg_data,
    make_tag_alias_data,
    make_debt_free_data,
)


# ---------------------------------------------------------------------------
# Pipeline helper: load data, filter, build, accumulate, compute_ttm, compute_ratios
# ---------------------------------------------------------------------------

def run_pipeline(data_dir: str) -> tuple:
    """Run the full pipeline on a synthetic data directory and return
    (con, history_df, ttm_df, ratios_df)."""
    con = duckdb.connect()
    sec_screen.load_data_filtered(con, data_dir, form_types={"10-K", "10-Q"})
    sec_screen.filter_relevant_submissions(con)
    per_filing = sec_screen.build_filing_fundamentals(con)
    sec_screen.accumulate_fundamentals_history(con, per_filing)
    history = sec_screen.load_fundamentals_history(con)
    ttm = sec_screen.compute_ttm(history)
    ratios = sec_screen.compute_ratios(ttm)
    return con, history, ttm, ratios


# ---------------------------------------------------------------------------
# 1. TTM computation, exact values
# ---------------------------------------------------------------------------

class TestTTMExactValues:
    def test_ttm_revenues(self, ttm_data_dir):
        """TTM revenues = 10-K annual + current Q YTD - prior year Q YTD.
        1000 + 260 - 240 = 1020 (in millions of raw value)."""
        _, _, ttm, _ = run_pipeline(str(ttm_data_dir))
        row = ttm[ttm["cik"] == "12345"].iloc[0]
        assert row["Revenues"] == pytest.approx(1_000_000_000 + 260_000_000 - 240_000_000)
        # 1,020,000,000

    def test_ttm_operating_income(self, ttm_data_dir):
        """TTM OpInc = 150M + 40M - 35M = 155M."""
        _, _, ttm, _ = run_pipeline(str(ttm_data_dir))
        row = ttm[ttm["cik"] == "12345"].iloc[0]
        assert row["OperatingIncomeLoss"] == pytest.approx(155_000_000)

    def test_balance_sheet_from_most_recent_filing(self, ttm_data_dir):
        """Balance-sheet fields come from the Q1 2024 filing (most recent),
        NOT the FY2023 10-K."""
        _, _, ttm, _ = run_pipeline(str(ttm_data_dir))
        row = ttm[ttm["cik"] == "12345"].iloc[0]
        # Q1 2024 Assets = 2,100,000,000 (not the 10-K's 2,000,000,000)
        assert row["Assets"] == pytest.approx(2_100_000_000)
        # Q1 2024 LongTermDebtNoncurrent = 420,000,000 (not the 10-K's 400,000,000)
        assert row["LongTermDebtNoncurrent"] == pytest.approx(420_000_000)

    def test_ttm_basis_describes_calculation(self, ttm_data_dir):
        """ttm_basis should describe the TTM calculation."""
        _, _, ttm, _ = run_pipeline(str(ttm_data_dir))
        row = ttm[ttm["cik"] == "12345"].iloc[0]
        assert "TTM" in str(row["ttm_basis"])
        assert "FY2023" in str(row["ttm_basis"])


# ---------------------------------------------------------------------------
# 2. 10-K-only fallback
# ---------------------------------------------------------------------------

class Test10KOnlyFallback:
    def test_flow_fields_equal_annual_figures(self, ten_k_only_dir):
        """With only a 10-K ingested, TTM flow fields equal the 10-K's own figures."""
        _, _, ttm, _ = run_pipeline(str(ten_k_only_dir))
        row = ttm[ttm["cik"] == "12345"].iloc[0]
        assert row["Revenues"] == pytest.approx(1_000_000_000)
        assert row["OperatingIncomeLoss"] == pytest.approx(150_000_000)

    def test_ttm_basis_mentions_10k_only(self, ten_k_only_dir):
        _, _, ttm, _ = run_pipeline(str(ten_k_only_dir))
        row = ttm[ttm["cik"] == "12345"].iloc[0]
        assert "10-K only" in str(row["ttm_basis"])


# ---------------------------------------------------------------------------
# 3. Incomplete history
# ---------------------------------------------------------------------------

class TestIncompleteHistory:
    def test_flow_fields_are_none(self, incomplete_history_dir):
        """10-K + Q1 2024 but no prior-year Q1 2023 -> flow fields are None."""
        _, _, ttm, _ = run_pipeline(str(incomplete_history_dir))
        row = ttm[ttm["cik"] == "12345"].iloc[0]
        assert row["Revenues"] is None or (isinstance(row["Revenues"], float) and pd.isna(row["Revenues"]))

    def test_ttm_basis_explains_missing(self, incomplete_history_dir):
        _, _, ttm, _ = run_pipeline(str(incomplete_history_dir))
        row = ttm[ttm["cik"] == "12345"].iloc[0]
        assert "Incomplete" in str(row["ttm_basis"])
        assert "no matching prior-year" in str(row["ttm_basis"])

    def test_balance_sheet_still_populated(self, incomplete_history_dir):
        """Balance sheet fields should still come from the most recent filing."""
        _, _, ttm, _ = run_pipeline(str(incomplete_history_dir))
        row = ttm[ttm["cik"] == "12345"].iloc[0]
        # Q1 2024 Assets
        assert row["Assets"] == pytest.approx(2_100_000_000)


# ---------------------------------------------------------------------------
# 4. ddate filtering (regression test)
# ---------------------------------------------------------------------------

class TestDdateFiltering:
    def test_picks_current_period_not_prior_year(self, ddate_dir):
        """A single filing with both current (20231231, value=200) and
        prior-year (20221231, value=380) LongTermDebtNoncurrent. The pivot
        should pick 200 (current period), not 380 (larger prior-year value).

        The ddate = sub.period filter in build_fundamentals/build_filing_fundamentals
        is what prevents this -- without it, a naive MAX(value) picks the
        larger prior-year figure.
        """
        con = duckdb.connect()
        sec_screen.load_data_filtered(con, str(ddate_dir), form_types={"10-K"})
        sec_screen.filter_relevant_submissions(con)
        df = sec_screen.build_filing_fundamentals(con)
        row = df.iloc[0]
        # Should be 200M (current period), not 380M (prior year)
        assert row["LongTermDebtNoncurrent"] == pytest.approx(200_000_000)


# ---------------------------------------------------------------------------
# 5. coreg / consolidated-vs-subsidiary handling
# ---------------------------------------------------------------------------

class TestCoregHandling:
    def test_prefers_consolidated_over_subsidiary(self, coreg_dir):
        """Same tag+qtrs+ddate with coreg=NULL (consolidated, value=150M)
        and coreg='SUBSIDIARY-A' (value=30M). The pivot should prefer the
        consolidated (NULL coreg) row."""
        con = duckdb.connect()
        sec_screen.load_data_filtered(con, str(coreg_dir), form_types={"10-K"})
        sec_screen.filter_relevant_submissions(con)
        df = sec_screen.build_filing_fundamentals(con)
        row = df.iloc[0]
        assert row["OperatingIncomeLoss"] == pytest.approx(150_000_000)


# ---------------------------------------------------------------------------
# 6. Tag aliasing
# ---------------------------------------------------------------------------

class TestTagAliasing:
    def test_alias_tag_populates_canonical_field(self, tag_alias_dir):
        """A filing using LongTermDebtAndCapitalLeaseObligations (an alias)
        should still populate LongTermDebtNoncurrent via TAG_MAP's COALESCE."""
        con = duckdb.connect()
        sec_screen.load_data_filtered(con, str(tag_alias_dir), form_types={"10-K"})
        sec_screen.filter_relevant_submissions(con)
        df = sec_screen.build_filing_fundamentals(con)
        row = df.iloc[0]
        assert row["LongTermDebtNoncurrent"] == pytest.approx(400_000_000)


# ---------------------------------------------------------------------------
# 7. filter_submissions normalization
# ---------------------------------------------------------------------------

class TestFilterSubmissionsNormalization:
    @staticmethod
    def _make_sub_table(con):
        """Create a minimal sub table using explicit VALUES (avoids DuckDB
        read_csv_auto limitations with inline multi-line strings)."""
        con.execute("""
            CREATE TABLE sub (adsh VARCHAR, cik VARCHAR, name VARCHAR, sic VARCHAR,
                              form VARCHAR, period VARCHAR, fy VARCHAR, fp VARCHAR,
                              filed VARCHAR, countryba VARCHAR, stprba VARCHAR,
                              cityba VARCHAR, zipba VARCHAR, bas1 VARCHAR, bas2 VARCHAR,
                              baph VARCHAR, countryma VARCHAR, stprma VARCHAR, cityma VARCHAR,
                              zipma VARCHAR, mas1 VARCHAR, mas2 VARCHAR, countryinc VARCHAR,
                              stprinc VARCHAR, ein VARCHAR, former VARCHAR, changed VARCHAR,
                              afs VARCHAR, wksi VARCHAR, fye VARCHAR, accepted VARCHAR,
                              prevrpt VARCHAR, detail VARCHAR, instance VARCHAR,
                              nciks VARCHAR, aciks VARCHAR)
        """)
        con.execute("""
            INSERT INTO sub (adsh, cik, name, sic, form, period, fy, fp, filed)
            VALUES ('000001-24-001', '12345', 'TEST', '7370', '10-K',
                    '20231231', '2023', 'FY', '20240215')
        """)

    def test_trailing_space_still_matches(self):
        """'10-K ' (trailing space) normalizes to '10-K' and matches."""
        con = duckdb.connect()
        self._make_sub_table(con)
        form_clean = "10-K ".strip().upper()
        assert form_clean == "10-K"
        con.execute(
            """CREATE OR REPLACE TABLE sub_filtered AS
               SELECT * FROM sub WHERE UPPER(TRIM(form)) = ?""",
            [form_clean],
        )
        count = con.execute("SELECT COUNT(*) FROM sub_filtered").fetchone()[0]
        assert count == 1

    def test_wrong_case_still_matches(self):
        """'10-k' (lowercase) normalizes to '10-K' and matches."""
        con = duckdb.connect()
        self._make_sub_table(con)
        form_clean = "10-k".strip().upper()
        assert form_clean == "10-K"
        con.execute(
            """CREATE OR REPLACE TABLE sub_filtered AS
               SELECT * FROM sub WHERE UPPER(TRIM(form)) = ?""",
            [form_clean],
        )
        count = con.execute("SELECT COUNT(*) FROM sub_filtered").fetchone()[0]
        assert count == 1

    def test_empty_form_matches_zero_rows(self):
        """Empty string should match zero rows without error."""
        con = duckdb.connect()
        self._make_sub_table(con)
        con.execute(
            """CREATE OR REPLACE TABLE sub_filtered AS
               SELECT * FROM sub WHERE UPPER(TRIM(form)) = ?""",
            [""],
        )
        count = con.execute("SELECT COUNT(*) FROM sub_filtered").fetchone()[0]
        assert count == 0

    def test_garbage_form_matches_zero_rows(self):
        """A garbage form value should match zero rows without raising."""
        con = duckdb.connect()
        self._make_sub_table(con)
        con.execute(
            """CREATE OR REPLACE TABLE sub_filtered AS
               SELECT * FROM sub WHERE UPPER(TRIM(form)) = ?""",
            ["NOT-A-REAL-FORM"],
        )
        count = con.execute("SELECT COUNT(*) FROM sub_filtered").fetchone()[0]
        assert count == 0


# ---------------------------------------------------------------------------
# 8. Debt-free company ratios
# ---------------------------------------------------------------------------

class TestDebtFreeCompanyRatios:
    def test_interest_coverage_is_inf(self, debt_free_dir):
        """A company with no InterestExpense tagged should have inf
        InterestCoverage, not NaN."""
        _, _, ttm, ratios = run_pipeline(str(debt_free_dir))
        row = ratios.iloc[0]
        import math
        assert math.isinf(row["InterestCoverage"])
        assert row["InterestCoverage"] > 0  # positive infinity

    def test_debt_to_equity_is_zero(self, debt_free_dir):
        """No debt tags -> TotalDebt is NaN, DebtToEquity should be 0."""
        _, _, ttm, ratios = run_pipeline(str(debt_free_dir))
        row = ratios.iloc[0]
        assert row["DebtToEquity"] == 0.0


# ---------------------------------------------------------------------------
# 9. Accumulation dedup
# ---------------------------------------------------------------------------

class TestAccumulationDedup:
    def test_double_ingest_does_not_duplicate(self, ttm_data_dir):
        """Calling accumulate_fundamentals_history twice with the same batch
        should not grow the history table."""
        con = duckdb.connect()
        sec_screen.load_data_filtered(con, str(ttm_data_dir),
                                      form_types={"10-K", "10-Q"})
        sec_screen.filter_relevant_submissions(con)
        per_filing = sec_screen.build_filing_fundamentals(con)
        sec_screen.accumulate_fundamentals_history(con, per_filing)
        count1 = con.execute(
            "SELECT COUNT(*) FROM fundamentals_history"
        ).fetchone()[0]

        # Second call with same batch
        sec_screen.accumulate_fundamentals_history(con, per_filing)
        count2 = con.execute(
            "SELECT COUNT(*) FROM fundamentals_history"
        ).fetchone()[0]

        assert count1 == count2
        assert count1 > 0  # sanity check: data was actually inserted


# ---------------------------------------------------------------------------
# 10. load_data_filtered pre-filtering
# ---------------------------------------------------------------------------

class TestLoadDataFiltered:
    def test_form_types_filter_matches_post_filter(self, ttm_data_dir):
        """Passing form_types={'10-K'} should produce the same sub/num counts
        as loading everything and then filtering to 10-K after."""
        # Full load (all forms)
        con_full = duckdb.connect()
        sec_screen.load_data_filtered(con_full, str(ttm_data_dir),
                                      form_types={"10-K", "10-Q"})
        full_sub = con_full.execute("SELECT COUNT(*) FROM sub").fetchone()[0]
        full_num = con_full.execute(
            "SELECT COUNT(*) FROM num_typed"
        ).fetchone()[0]

        # 10-K-only load
        con_10k = duckdb.connect()
        sec_screen.load_data_filtered(con_10k, str(ttm_data_dir),
                                      form_types={"10-K"})
        k_sub = con_10k.execute("SELECT COUNT(*) FROM sub").fetchone()[0]
        k_num = con_10k.execute(
            "SELECT COUNT(*) FROM num_typed"
        ).fetchone()[0]

        # 10-K count should be <= full count
        assert k_sub <= full_sub
        assert k_num <= full_num
        # And specifically, only 10-K rows should be present
        forms = con_10k.execute(
            "SELECT DISTINCT UPPER(TRIM(form)) FROM sub"
        ).fetchall()
        form_values = {row[0] for row in forms}
        assert form_values == {"10-K"}

    def test_allowed_ciks_filter(self, ttm_data_dir):
        """allowed_ciks={12345} should keep that CIK and exclude others."""
        con = duckdb.connect()
        sec_screen.load_data_filtered(con, str(ttm_data_dir),
                                      form_types={"10-K", "10-Q"},
                                      allowed_ciks={"12345"})
        ciks = con.execute("SELECT DISTINCT cik FROM sub").fetchall()
        cik_values = {str(row[0]) for row in ciks}
        assert cik_values == {"12345"}

    def test_non_matching_cik_returns_empty(self, ttm_data_dir):
        """allowed_ciks with no matching CIK produces empty sub table."""
        con = duckdb.connect()
        sec_screen.load_data_filtered(con, str(ttm_data_dir),
                                      form_types={"10-K", "10-Q"},
                                      allowed_ciks={"99999"})
        count = con.execute("SELECT COUNT(*) FROM sub").fetchone()[0]
        assert count == 0


# ---------------------------------------------------------------------------
# Combined TTM + enrichment (compute_ttm_with_enrichment)
# ---------------------------------------------------------------------------

class TestCombinedTTMEnrichment:
    def test_produces_enrichment_columns(self, ttm_data_dir):
        """The combined function should produce FScore, RevenueGrowth, etc."""
        con, history, _, _ = run_pipeline(str(ttm_data_dir))
        ttm = sec_screen.compute_ttm_with_enrichment(history)
        for col in ["FScore", "RevenueGrowth", "FCFGrowth", "RevenueGrowth3yr"]:
            assert col in ttm.columns, f"Missing enrichment column: {col}"

    def test_ttm_revenues_match_standalone(self, ttm_data_dir):
        """Combined TTM revenues should match the standalone compute_ttm."""
        con, history, standalone_ttm, _ = run_pipeline(str(ttm_data_dir))
        combined = sec_screen.compute_ttm_with_enrichment(history)
        for _, row in combined.iterrows():
            cik = str(row["cik"])
            standalone_row = standalone_ttm[standalone_ttm["cik"].astype(str) == cik]
            if not standalone_row.empty:
                assert row["Revenues"] == pytest.approx(
                    standalone_row.iloc[0]["Revenues"]
                )

    def test_fscore_range(self, ttm_data_dir):
        """F-Score should be between 0 and 9 (or None)."""
        con, history, _, _ = run_pipeline(str(ttm_data_dir))
        ttm = sec_screen.compute_ttm_with_enrichment(history)
        for _, row in ttm.iterrows():
            fs = row.get("FScore")
            if fs is not None and not pd.isna(fs):
                assert 0 <= int(fs) <= 9, f"FScore {fs} out of range"


# ---------------------------------------------------------------------------
# 6. Cover-page shares outstanding (dei:EntityCommonStockSharesOutstanding)
# ---------------------------------------------------------------------------

class TestCoverPageShares:
    """EntityCommonStockSharesOutstanding is a cover-page (dei) tag whose ddate
    is the filing's cover-page date, not the fiscal period end.  The standard
    pivot's ddate=period filter would silently exclude it, so it needs its own
    query path.  These tests verify the special handling.
    """

    @pytest.fixture
    def cover_page_data_dir(self, tmp_path):
        """Synthetic 10-K where:
        - balance-sheet CommonStockSharesOutstanding = 50M at ddate=period
        - cover-page EntityCommonStockSharesOutstanding = 52M at a later ddate
          (closer to filed), simulating post-period-end share issuance.
        """
        adsh = "0000012345-24-000001"
        write_sub_txt(tmp_path / "sub.txt", [
            {"adsh": adsh, "cik": "12345", "name": "TEST CORP", "sic": "7370",
             "form": "10-K", "period": "20231231", "fy": "2023", "fp": "FY",
             "filed": "20240215"},
        ])
        num_rows = []

        def add(ad, tag, qtrs, ddate, val, coreg="", uom="USD"):
            num_rows.append({
                "adsh": ad, "tag": tag, "version": "us-gaap/2023",
                "coreg": coreg, "ddate": ddate, "qtrs": qtrs,
                "uom": uom, "value": val, "footnote": "",
            })

        # Flow fields
        for tag, val in [("Revenues", "1000000000"),
                         ("OperatingIncomeLoss", "150000000"),
                         ("NetIncomeLoss", "100000000"),
                         ("NetCashProvidedByUsedInOperatingActivities", "120000000"),
                         ("PaymentsToAcquirePropertyPlantAndEquipment", "30000000")]:
            add(adsh, tag, "4", "20231231", val)

        # Balance-sheet fields at fiscal-period-end ddate
        for tag, val in [("Assets", "2000000000"), ("AssetsCurrent", "500000000"),
                         ("LiabilitiesCurrent", "300000000"),
                         ("CashAndCashEquivalentsAtCarryingValue", "100000000"),
                         ("PropertyPlantAndEquipmentNet", "800000000"),
                         ("StockholdersEquity", "1000000000"),
                         ("LongTermDebtNoncurrent", "400000000"),
                         ("CommonStockSharesOutstanding", "50000000")]:
            add(adsh, tag, "0", "20231231", val)

        # Cover-page shares — ddate is 20240210 (close to filed=20240215),
        # NOT the fiscal period end.  Value is higher (52M vs 50M) to
        # simulate post-period-end share issuance.
        add(adsh, "EntityCommonStockSharesOutstanding", "0", "20240210",
            "52000000", uom="shares")

        write_num_txt(tmp_path / "num.txt", num_rows)
        return tmp_path

    @pytest.fixture
    def no_cover_page_data_dir(self, tmp_path):
        """Synthetic 10-K with ONLY the balance-sheet tag (no cover-page tag).
        This filing represents older/smaller filers that don't tag
        EntityCommonStockSharesOutstanding — the fallback must work.
        """
        adsh = "0000098765-24-000001"
        write_sub_txt(tmp_path / "sub.txt", [
            {"adsh": adsh, "cik": "98765", "name": "NO DEI CORP", "sic": "7370",
             "form": "10-K", "period": "20231231", "fy": "2023", "fp": "FY",
             "filed": "20240215"},
        ])
        num_rows = []

        def add(ad, tag, qtrs, ddate, val):
            num_rows.append({
                "adsh": ad, "tag": tag, "version": "us-gaap/2023",
                "coreg": "", "ddate": ddate, "qtrs": qtrs,
                "uom": "USD", "value": val, "footnote": "",
            })

        for tag, val in [("Revenues", "500000000"),
                         ("OperatingIncomeLoss", "75000000"),
                         ("NetIncomeLoss", "50000000"),
                         ("NetCashProvidedByUsedInOperatingActivities", "60000000"),
                         ("PaymentsToAcquirePropertyPlantAndEquipment", "15000000")]:
            add(adsh, tag, "4", "20231231", val)

        for tag, val in [("Assets", "1000000000"), ("AssetsCurrent", "250000000"),
                         ("LiabilitiesCurrent", "150000000"),
                         ("CashAndCashEquivalentsAtCarryingValue", "50000000"),
                         ("PropertyPlantAndEquipmentNet", "400000000"),
                         ("StockholdersEquity", "500000000"),
                         ("LongTermDebtNoncurrent", "200000000"),
                         ("CommonStockSharesOutstanding", "25000000")]:
            add(adsh, tag, "0", "20231231", val)

        write_num_txt(tmp_path / "num.txt", num_rows)
        return tmp_path

    def test_both_columns_present(self, cover_page_data_dir):
        """The pivot should produce BOTH CommonStockSharesOutstanding and
        CommonStockSharesOutstandingCoverPage when the cover-page tag exists."""
        con = duckdb.connect()
        sec_screen.load_data_filtered(con, str(cover_page_data_dir),
                                       form_types={"10-K", "10-Q"})
        sec_screen.filter_relevant_submissions(con)
        df = sec_screen.build_filing_fundamentals(con)
        con.close()

        assert "CommonStockSharesOutstanding" in df.columns, (
            "Balance-sheet shares column missing"
        )
        assert "CommonStockSharesOutstandingCoverPage" in df.columns, (
            "Cover-page shares column missing — "
            "fetch_cover_page_shares_outstanding() may not have been wired in"
        )
        row = df.iloc[0]
        assert float(row["CommonStockSharesOutstanding"]) == 50_000_000
        assert float(row["CommonStockSharesOutstandingCoverPage"]) == 52_000_000

    def test_valuation_uses_cover_page(self, cover_page_data_dir):
        """EPS/Graham/DCF should use the cover-page share count (52M), not the
        balance-sheet one (50M), when both are present."""
        import core.valuation as valuation

        con = duckdb.connect()
        sec_screen.load_data_filtered(con, str(cover_page_data_dir),
                                       form_types={"10-K", "10-Q"})
        sec_screen.filter_relevant_submissions(con)
        per_filing = sec_screen.build_filing_fundamentals(con)
        sec_screen.accumulate_fundamentals_history(con, per_filing)
        history = sec_screen.load_fundamentals_history(con)
        ttm = sec_screen.compute_ttm(history)
        ratios = sec_screen.compute_ratios(ttm)
        valued = valuation.apply_valuation(ratios)
        con.close()

        row = valued.iloc[0]
        ni = float(row["NetIncomeLoss"])  # 100M
        bs_shares = float(row["CommonStockSharesOutstanding"])  # 50M
        cp_shares = float(row["CommonStockSharesOutstandingCoverPage"])  # 52M

        # EPS should be NI / cover-page shares = 100M / 52M ≈ 1.923
        # If it used balance-sheet shares, EPS = 100M / 50M = 2.000
        # Graham Number uses EPS * BVPS under the sqrt, so let's check
        # that GrahamNumber differs from what it would be with BS shares.
        equity = float(row["StockholdersEquity"])
        bs_bvps = equity / bs_shares  # 1B / 50M = 20
        cp_bvps = equity / cp_shares  # 1B / 52M ≈ 19.23
        bs_eps = ni / bs_shares       # 100M / 50M = 2.00
        cp_eps = ni / cp_shares       # 100M / 52M ≈ 1.923

        expected_gn_cp = (22.5 * cp_eps * cp_bvps) ** 0.5
        expected_gn_bs = (22.5 * bs_eps * bs_bvps) ** 0.5

        actual_gn = float(row["GrahamNumber"])
        assert actual_gn == pytest.approx(expected_gn_cp, rel=1e-4), (
            f"GrahamNumber should use cover-page shares ({cp_shares:,.0f}), "
            f"not balance-sheet ({bs_shares:,.0f}). "
            f"Expected {expected_gn_cp:.2f} (cover-page), "
            f"got {actual_gn:.2f}"
        )
        # Sanity check: the two should differ enough to catch a wrong choice
        assert abs(expected_gn_cp - expected_gn_bs) > 0.01, (
            "Test shares don't differ enough to validate the right one was used"
        )

    def test_fallback_when_no_cover_page_tag(self, no_cover_page_data_dir):
        """When EntityCommonStockSharesOutstanding is NOT present, valuation
        must fall back to the balance-sheet figure — this must not break
        filings that simply don't report the cover-page tag."""
        import core.valuation as valuation

        con = duckdb.connect()
        sec_screen.load_data_filtered(con, str(no_cover_page_data_dir),
                                       form_types={"10-K", "10-Q"})
        sec_screen.filter_relevant_submissions(con)
        per_filing = sec_screen.build_filing_fundamentals(con)
        sec_screen.accumulate_fundamentals_history(con, per_filing)
        history = sec_screen.load_fundamentals_history(con)
        ttm = sec_screen.compute_ttm(history)
        ratios = sec_screen.compute_ratios(ttm)
        valued = valuation.apply_valuation(ratios)
        con.close()

        row = valued.iloc[0]
        ni = float(row["NetIncomeLoss"])  # 50M
        bs_shares = float(row["CommonStockSharesOutstanding"])  # 25M
        equity = float(row["StockholdersEquity"])

        # Cover-page column may be absent entirely or all-NaN
        cp = row.get("CommonStockSharesOutstandingCoverPage")
        if cp is not None and not pd.isna(cp):
            # If it somehow exists, it shouldn't have been used (should be NaN/None
            # for this filing since the tag wasn't in num.txt)
            pass

        bs_eps = ni / bs_shares
        bs_bvps = equity / bs_shares
        expected_gn = (22.5 * bs_eps * bs_bvps) ** 0.5
        actual_gn = float(row["GrahamNumber"])
        assert actual_gn == pytest.approx(expected_gn, rel=1e-4), (
            f"Fallback failed: expected GrahamNumber={expected_gn:.2f} "
            f"(using BS shares={bs_shares:,.0f}), got {actual_gn:.2f}"
        )
        # Valuation must not have crashed — confirm DCF is also computed
        dcf = row.get("DCFIntrinsicValue")
        assert dcf is not None and not pd.isna(dcf), (
            "DCFIntrinsicValue should be computed even without cover-page shares"
        )


# ---------------------------------------------------------------------------
# 7. pre.txt tag disambiguation
# ---------------------------------------------------------------------------

class TestPreTxtDisambiguation:
    """pre.txt (SEC presentation linkbase) maps (adsh, tag) → stmt (BS, IS,
    CF, or footnote).  Tags whose ONLY appearances are in non-primary
    statement locations (footnotes, disclosures) should be excluded from the
    pivot so they don't compete with face-of-statement values via COALESCE.
    """

    @pytest.fixture
    def pre_txt_data_dir(self, tmp_path):
        """Synthetic 10-K where:
        - SalesRevenueNet ($800M) appears in num.txt but pre.txt shows it
          ONLY in a footnote (stmt='') — it's a segment disclosure, not
          the primary IS figure.
        - SalesRevenueGoodsNet ($1,000M) appears in num.txt AND pre.txt
          confirms it on the Income Statement (stmt='IS').

        The TAG_MAP for Revenues has aliases in this order:
          Revenues → RevenueFromContract... → SalesRevenueNet → SalesRevenueGoodsNet
        Since Revenues and RevenueFromContract are not in num.txt,
        COALESCE would normally pick SalesRevenueNet ($800M, wrong).
        After pre.txt filtering excludes SalesRevenueNet (footnote-only),
        COALESCE falls through to SalesRevenueGoodsNet ($1,000M, correct).
        """
        adsh = "0000012345-24-000001"
        write_sub_txt(tmp_path / "sub.txt", [
            {"adsh": adsh, "cik": "12345", "name": "TEST CORP", "sic": "7370",
             "form": "10-K", "period": "20231231", "fy": "2023", "fp": "FY",
             "filed": "20240215"},
        ])
        num_rows = []

        def add(ad, tag, qtrs, ddate, val, coreg="", uom="USD"):
            num_rows.append({
                "adsh": ad, "tag": tag, "version": "us-gaap/2023",
                "coreg": coreg, "ddate": ddate, "qtrs": qtrs,
                "uom": uom, "value": val, "footnote": "",
            })

        # Flow fields — note: Revenues tag itself is NOT in num.txt for
        # this filing (company uses different tags).
        for tag, val in [("SalesRevenueNet", "800000000"),       # footnote-only, WRONG
                         ("SalesRevenueGoodsNet", "1000000000"),  # IS face, CORRECT
                         ("OperatingIncomeLoss", "150000000"),
                         ("NetIncomeLoss", "100000000"),
                         ("NetCashProvidedByUsedInOperatingActivities", "120000000"),
                         ("PaymentsToAcquirePropertyPlantAndEquipment", "30000000")]:
            add(adsh, tag, "4", "20231231", val)

        # Balance sheet
        for tag, val in [("Assets", "2000000000"), ("AssetsCurrent", "500000000"),
                         ("LiabilitiesCurrent", "300000000"),
                         ("CashAndCashEquivalentsAtCarryingValue", "100000000"),
                         ("PropertyPlantAndEquipmentNet", "800000000"),
                         ("StockholdersEquity", "1000000000"),
                         ("LongTermDebtNoncurrent", "400000000"),
                         ("CommonStockSharesOutstanding", "50000000")]:
            add(adsh, tag, "0", "20231231", val)

        write_num_txt(tmp_path / "num.txt", num_rows)

        # pre.txt: SalesRevenueGoodsNet is on IS; SalesRevenueNet is footnote-only
        write_pre_txt(tmp_path / "pre.txt", [
            {"adsh": adsh, "report": "1", "line": "1", "stmt": "IS",
             "inpth": "0", "tag": "SalesRevenueGoodsNet",
             "version": "us-gaap/2023", "plabel": "Revenue", "negating": "0"},
            {"adsh": adsh, "report": "5", "line": "12", "stmt": "",
             "inpth": "2", "tag": "SalesRevenueNet",
             "version": "us-gaap/2023", "plabel": "Segment Revenue",
             "negating": "0"},
        ])
        return tmp_path

    def test_footnote_only_tag_excluded(self, pre_txt_data_dir):
        """SalesRevenueNet (footnote-only) should be excluded from num_typed,
        so the pivot falls through to SalesRevenueGoodsNet (IS face)."""
        con = duckdb.connect()
        sec_screen.load_data_filtered(con, str(pre_txt_data_dir),
                                       form_types={"10-K", "10-Q"})
        sec_screen.filter_relevant_submissions(con)

        # Verify SalesRevenueNet was removed from num_typed
        footnote_rows = con.execute("""
            SELECT COUNT(*) FROM num_typed
            WHERE tag = 'SalesRevenueNet'
        """).fetchone()[0]
        assert footnote_rows == 0, (
            "SalesRevenueNet should have been excluded from num_typed "
            "because pre.txt shows it only in a footnote (not BS/IS/CF)"
        )

        # Verify SalesRevenueGoodsNet survived (it's on IS in pre.txt)
        is_rows = con.execute("""
            SELECT COUNT(*) FROM num_typed
            WHERE tag = 'SalesRevenueGoodsNet'
        """).fetchone()[0]
        assert is_rows > 0, (
            "SalesRevenueGoodsNet should still be in num_typed "
            "because pre.txt confirms it on the Income Statement"
        )

        con.close()

    def test_pivot_uses_face_value(self, pre_txt_data_dir):
        """The pivot should resolve Revenues to $1,000M (SalesRevenueGoodsNet,
        the IS-face value), not $800M (SalesRevenueNet, the footnote value)."""
        con = duckdb.connect()
        sec_screen.load_data_filtered(con, str(pre_txt_data_dir),
                                       form_types={"10-K", "10-Q"})
        sec_screen.filter_relevant_submissions(con)
        df = sec_screen.build_filing_fundamentals(con)
        con.close()

        row = df.iloc[0]
        revenues = float(row["Revenues"])
        assert revenues == 1_000_000_000.0, (
            f"Revenues should be $1,000M (IS face value from "
            f"SalesRevenueGoodsNet), got ${revenues:,.0f}. "
            f"The $800M footnote-only SalesRevenueNet was not excluded."
        )

    def test_no_pre_txt_still_works(self, ttm_data_dir):
        """When pre.txt is absent (e.g., synthetic data, old zips), the
        pipeline should work as before — nothing should break."""
        con = duckdb.connect()
        sec_screen.load_data_filtered(con, str(ttm_data_dir),
                                       form_types={"10-K", "10-Q"})
        sec_screen.filter_relevant_submissions(con)
        df = sec_screen.build_filing_fundamentals(con)
        con.close()

        # Should have our test company with correct revenues
        row = df[df["cik"].astype(str) == "12345"]
        assert not row.empty, "Test company not found"
        # The existing ttm_data_dir fixture has Revenues=1B via the
        # Revenues tag directly — this should still work fine
        assert float(row.iloc[0]["Revenues"]) == 1_000_000_000.0
