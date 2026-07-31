"""Tests for core/insider_analysis.py."""

import io
import os
import zipfile
from pathlib import Path

import pandas as pd
import pytest

import core.insider_analysis as ia


# ---------------------------------------------------------------------------
# Helpers — build synthetic SEC-format ZIP files
# ---------------------------------------------------------------------------

def _make_zip(zip_path: Path, trans_rows, sub_rows, owner_rows):
    """Write a synthetic quarterly ZIP with the three required SEC files."""
    with zipfile.ZipFile(zip_path, "w") as zf:
        # NONDERIV_TRANS.txt
        trans_cols = [
            "ACCESSION_NUMBER", "TRANS_DATE", "TRANSACTION_CODE",
            "TRANS_SHARES", "TRANS_PRICEPERSHARE", "TRANS_ACQUIRED_DISP_CD",
        ]
        buf = io.StringIO()
        buf.write("\t".join(trans_cols) + "\n")
        for r in trans_rows:
            buf.write("\t".join(str(r.get(c, "")) for c in trans_cols) + "\n")
        zf.writestr("NONDERIV_TRANS.txt", buf.getvalue())

        # SUBMISSION.txt
        sub_cols = [
            "ACCESSION_NUMBER", "ISSUERCIK", "ISSUERTRADINGSYMBOL", "PERIODOFREPORT",
        ]
        buf = io.StringIO()
        buf.write("\t".join(sub_cols) + "\n")
        for r in sub_rows:
            buf.write("\t".join(str(r.get(c, "")) for c in sub_cols) + "\n")
        zf.writestr("SUBMISSION.txt", buf.getvalue())

        # REPORTINGOWNER.txt
        own_cols = [
            "ACCESSION_NUMBER", "RPTOWNERCIK", "RPTOWNERNAME",
            "RPTOWNERRELATIONSHIP", "RPTOWNERTITLE",
        ]
        buf = io.StringIO()
        buf.write("\t".join(own_cols) + "\n")
        for r in owner_rows:
            buf.write("\t".join(str(r.get(c, "")) for c in own_cols) + "\n")
        zf.writestr("REPORTINGOWNER.txt", buf.getvalue())


# ---------------------------------------------------------------------------
# 1. load_and_process_data()
# ---------------------------------------------------------------------------

class TestLoadAndProcessData:
    def test_basic_ingestion(self, tmp_path):
        """Single quarter with 2 trades should produce 2 rows."""
        acc1, acc2 = "000001-24-001", "000001-24-002"
        _make_zip(
            tmp_path / "insider_2024q1.zip",
            trans_rows=[
                {"ACCESSION_NUMBER": acc1, "TRANS_DATE": "15-JAN-2024",
                 "TRANSACTION_CODE": "P", "TRANS_SHARES": "1000",
                 "TRANS_PRICEPERSHARE": "50.00", "TRANS_ACQUIRED_DISP_CD": "A"},
                {"ACCESSION_NUMBER": acc2, "TRANS_DATE": "20-JAN-2024",
                 "TRANSACTION_CODE": "S", "TRANS_SHARES": "500",
                 "TRANS_PRICEPERSHARE": "52.00", "TRANS_ACQUIRED_DISP_CD": "D"},
            ],
            sub_rows=[
                {"ACCESSION_NUMBER": acc1, "ISSUERCIK": "12345",
                 "ISSUERTRADINGSYMBOL": "ABC", "PERIODOFREPORT": "20240131"},
                {"ACCESSION_NUMBER": acc2, "ISSUERCIK": "12345",
                 "ISSUERTRADINGSYMBOL": "ABC", "PERIODOFREPORT": "20240131"},
            ],
            owner_rows=[
                {"ACCESSION_NUMBER": acc1, "RPTOWNERCIK": "99999",
                 "RPTOWNERNAME": "JOHN DOE", "RPTOWNERRELATIONSHIP": "CEO",
                 "RPTOWNERTITLE": "Chief Executive Officer"},
                {"ACCESSION_NUMBER": acc2, "RPTOWNERCIK": "99999",
                 "RPTOWNERNAME": "JOHN DOE", "RPTOWNERRELATIONSHIP": "CEO",
                 "RPTOWNERTITLE": "Chief Executive Officer"},
            ],
        )

        df = ia.load_and_process_data(str(tmp_path), start_year=2024, end_year=2024)
        assert len(df) == 2
        assert df["TRANSACTION_CODE"].iloc[0] == "P"
        assert df["TRANS_SHARES"].iloc[0] == 1000.0
        assert df["QUARTER"].iloc[0] == "2024q1"
        assert df["ISSUERCIK"].iloc[0] == "12345"
        assert df["RPTOWNERCIK"].iloc[0] == "99999"

    def test_filters_non_open_market(self, tmp_path):
        """Option exercises (code 'M') and grants ('A') should be excluded."""
        acc = "000001-24-001"
        _make_zip(
            tmp_path / "insider_2024q1.zip",
            trans_rows=[
                {"ACCESSION_NUMBER": acc, "TRANS_DATE": "15-JAN-2024",
                 "TRANSACTION_CODE": "P", "TRANS_SHARES": "1000",
                 "TRANS_PRICEPERSHARE": "50.00", "TRANS_ACQUIRED_DISP_CD": "A"},
                {"ACCESSION_NUMBER": acc + "x", "TRANS_DATE": "15-JAN-2024",
                 "TRANSACTION_CODE": "M", "TRANS_SHARES": "5000",  # exercise
                 "TRANS_PRICEPERSHARE": "25.00", "TRANS_ACQUIRED_DISP_CD": "A"},
            ],
            sub_rows=[
                {"ACCESSION_NUMBER": acc, "ISSUERCIK": "12345",
                 "ISSUERTRADINGSYMBOL": "ABC", "PERIODOFREPORT": ""},
                {"ACCESSION_NUMBER": acc + "x", "ISSUERCIK": "12345",
                 "ISSUERTRADINGSYMBOL": "ABC", "PERIODOFREPORT": ""},
            ],
            owner_rows=[
                {"ACCESSION_NUMBER": acc, "RPTOWNERCIK": "99999",
                 "RPTOWNERNAME": "JOHN DOE", "RPTOWNERRELATIONSHIP": "CEO",
                 "RPTOWNERTITLE": ""},
                {"ACCESSION_NUMBER": acc + "x", "RPTOWNERCIK": "99999",
                 "RPTOWNERNAME": "JOHN DOE", "RPTOWNERRELATIONSHIP": "CEO",
                 "RPTOWNERTITLE": ""},
            ],
        )

        df = ia.load_and_process_data(str(tmp_path), start_year=2024, end_year=2024)
        assert len(df) == 1, f"Expected 1 trade (P only), got {len(df)}"
        assert df["TRANSACTION_CODE"].iloc[0] == "P"

    def test_date_parsing(self, tmp_path):
        """Dates in DD-MON-YYYY format should parse to datetime64."""
        acc = "000001-24-001"
        _make_zip(
            tmp_path / "insider_2024q1.zip",
            trans_rows=[
                {"ACCESSION_NUMBER": acc, "TRANS_DATE": "31-DEC-2024",
                 "TRANSACTION_CODE": "P", "TRANS_SHARES": "100",
                 "TRANS_PRICEPERSHARE": "10.00", "TRANS_ACQUIRED_DISP_CD": "A"},
            ],
            sub_rows=[
                {"ACCESSION_NUMBER": acc, "ISSUERCIK": "12345",
                 "ISSUERTRADINGSYMBOL": "XYZ", "PERIODOFREPORT": ""},
            ],
            owner_rows=[
                {"ACCESSION_NUMBER": acc, "RPTOWNERCIK": "88888",
                 "RPTOWNERNAME": "JANE DOE", "RPTOWNERRELATIONSHIP": "CFO",
                 "RPTOWNERTITLE": ""},
            ],
        )

        df = ia.load_and_process_data(str(tmp_path), start_year=2024, end_year=2024)
        assert pd.api.types.is_datetime64_any_dtype(df["TRANS_DATE"])
        assert df["TRANS_DATE"].dt.year.iloc[0] == 2024
        assert df["TRANS_DATE"].dt.month.iloc[0] == 12

    def test_form345_column_names(self, tmp_path):
        """Form 3/4/5 data uses TRANSACTION_DATE, TRANSACTION_SHARES, etc.
        — these should be normalised to the canonical column names."""
        acc = "000001-24-001"
        # Build ZIP with Form 345 naming convention
        with zipfile.ZipFile(tmp_path / "2016q1_form345.zip", "w") as zf:
            # NONDERIV_TRANS.txt with Form 345 column names
            trans_cols = [
                "ACCESSION_NUMBER", "TRANSACTION_DATE", "TRANSACTION_CODE",
                "TRANSACTION_SHARES", "TRANSACTION_PRICE_PER_SHARE",
                "TRANSACTION_ACQUIRED_DISP_CD",
            ]
            buf = io.StringIO()
            buf.write("\t".join(trans_cols) + "\n")
            buf.write(f"{acc}\t15-JAN-2016\tP\t1000\t50.00\tA\n")
            zf.writestr("NONDERIV_TRANS.txt", buf.getvalue())

            # SUBMISSION.txt with Form 345 column names
            sub_cols = ["ACCESSION_NUMBER", "ISSUERCIK", "ISSUERTRADINGSYMBOL"]
            buf = io.StringIO()
            buf.write("\t".join(sub_cols) + "\n")
            buf.write(f"{acc}\t12345\tABC\n")
            zf.writestr("SUBMISSION.txt", buf.getvalue())

            # REPORTINGOWNER.txt with Form 345 column names
            own_cols = ["ACCESSION_NUMBER", "RPTOWNERCIK", "RPTOWNERNAME",
                        "RPTOWNERRELATIONSHIP", "RPTOWNERTITLE"]
            buf = io.StringIO()
            buf.write("\t".join(own_cols) + "\n")
            buf.write(f"{acc}\t99999\tJOHN DOE\tCEO\tChief Executive Officer\n")
            zf.writestr("REPORTINGOWNER.txt", buf.getvalue())

        df = ia.load_and_process_data(str(tmp_path), start_year=2016, end_year=2016)
        assert len(df) == 1
        assert df["TRANS_SHARES"].iloc[0] == 1000.0  # normalised from TRANSACTION_SHARES
        assert df["TRANS_PRICEPERSHARE"].iloc[0] == 50.0  # from TRANSACTION_PRICE_PER_SHARE
        assert df["ISSUERCIK"].iloc[0] == "12345"

    def test_missing_quarter_skipped(self, tmp_path):
        """Trades outside the year range are filtered out by trade date."""
        _make_zip(
            tmp_path / "insider_2020q1.zip",
            trans_rows=[
                {"ACCESSION_NUMBER": "x", "TRANS_DATE": "15-JAN-2020",
                 "TRANSACTION_CODE": "P", "TRANS_SHARES": "100",
                 "TRANS_PRICEPERSHARE": "10.00", "TRANS_ACQUIRED_DISP_CD": "A"},
            ],
            sub_rows=[
                {"ACCESSION_NUMBER": "x", "ISSUERCIK": "12345",
                 "ISSUERTRADINGSYMBOL": "ABC", "PERIODOFREPORT": ""},
            ],
            owner_rows=[
                {"ACCESSION_NUMBER": "x", "RPTOWNERCIK": "99999",
                 "RPTOWNERNAME": "NAME", "RPTOWNERRELATIONSHIP": "CEO",
                 "RPTOWNERTITLE": ""},
            ],
        )

        # ZIP is found and processed, but the 2020 trades are filtered
        # out by the year-2024 cutoff → ValueError
        with pytest.raises(ValueError, match="No valid transaction"):
            ia.load_and_process_data(str(tmp_path), start_year=2024, end_year=2024)


# ---------------------------------------------------------------------------
# 2. classify_insiders()
# ---------------------------------------------------------------------------

class TestClassifyInsiders:
    def _make_trade_df(self, rows):
        """Convenience: build a DataFrame from (cik, date, code, shares) tuples."""
        data = []
        for cik, date_str, code, shares in rows:
            data.append({
                "RPTOWNERCIK": cik,
                "TRANS_DATE": pd.Timestamp(date_str),
                "TRANSACTION_CODE": code,
                "TRANS_SHARES": shares,
            })
        return pd.DataFrame(data)

    def test_all_opportunistic_with_insufficient_history(self):
        """An insider with only 2 years of January trades is NOT routine."""
        df = self._make_trade_df([
            ("111", "2022-01-15", "P", 100),
            ("111", "2023-01-20", "P", 200),
            # Only 2 unique years in January → not routine
        ])
        result = ia.classify_insiders(df)
        assert (result["TRADE_TYPE"] == "OPPORTUNISTIC").all()
        assert (result["ROUTINE_YEARS"] <= 2).all()

    def test_routine_with_three_years(self):
        """3 unique years of January trades → ROUTINE."""
        df = self._make_trade_df([
            ("111", "2021-01-10", "P", 100),
            ("111", "2022-01-15", "S", 200),
            ("111", "2023-01-20", "P", 150),
            ("111", "2024-01-05", "P", 300),
            # 4 unique years in January → routine (in 2024)
        ])
        result = ia.classify_insiders(df)
        assert (result["TRADE_TYPE"] == "ROUTINE").all()

    def test_routine_only_in_specific_month(self):
        """Routine in Jan doesn't make you routine in Feb."""
        df = self._make_trade_df([
            ("111", "2021-01-10", "P", 100),
            ("111", "2022-01-15", "P", 200),
            ("111", "2023-01-20", "P", 150),
            # 3 unique years in Jan → ROUTINE for Jan
            ("111", "2024-02-05", "S", 50),
            # Only 1 year in Feb → OPPORTUNISTIC
        ])
        result = ia.classify_insiders(df)
        jan_trades = result[result["TRANS_DATE"].dt.month == 1]
        feb_trades = result[result["TRANS_DATE"].dt.month == 2]
        assert (jan_trades["TRADE_TYPE"] == "ROUTINE").all()
        assert (feb_trades["TRADE_TYPE"] == "OPPORTUNISTIC").all()

    def test_mixed_insiders(self):
        """Multiple insiders — some routine, some not."""
        df = self._make_trade_df([
            # Insider A: 3 Januaries → routine
            ("AAA", "2021-01-10", "P", 100),
            ("AAA", "2022-01-12", "P", 100),
            ("AAA", "2023-01-15", "P", 100),
            # Insider B: only 2 Januaries → opportunistic
            ("BBB", "2022-01-10", "P", 100),
            ("BBB", "2023-01-20", "P", 100),
        ])
        result = ia.classify_insiders(df)
        a = result[result["RPTOWNERCIK"] == "AAA"]
        b = result[result["RPTOWNERCIK"] == "BBB"]
        assert (a["TRADE_TYPE"] == "ROUTINE").all()
        assert (b["TRADE_TYPE"] == "OPPORTUNISTIC").all()

    def test_100_percent_coverage(self):
        """Every trade must be either ROUTINE or OPPORTUNISTIC."""
        df = self._make_trade_df([
            ("111", "2021-01-10", "P", 100),
            ("111", "2022-01-15", "P", 200),
            ("111", "2023-01-20", "P", 150),
            ("222", "2023-06-01", "S", 500),
        ])
        result = ia.classify_insiders(df)
        total = len(result)
        opp = (result["TRADE_TYPE"] == "OPPORTUNISTIC").sum()
        rout = (result["TRADE_TYPE"] == "ROUTINE").sum()
        assert opp + rout == total
        assert opp >= 0
        assert rout >= 0

    def test_empty_dataframe(self):
        """Empty input should return an empty DataFrame with expected columns."""
        df = pd.DataFrame(columns=["RPTOWNERCIK", "TRANS_DATE"])
        result = ia.classify_insiders(df)
        assert result.empty
        assert "TRADE_TYPE" in result.columns
        assert "ROUTINE_YEARS" in result.columns

    def test_cache_returns_same_result(self):
        """Calling classify_insiders twice should return identical results."""
        df = self._make_trade_df([
            ("111", "2021-01-10", "P", 100),
            ("111", "2022-01-15", "P", 200),
            ("111", "2023-01-20", "P", 150),
        ])
        r1 = ia.classify_insiders(df, use_cache=True)
        r2 = ia.classify_insiders(df, use_cache=True)
        assert r1["TRADE_TYPE"].equals(r2["TRADE_TYPE"])
        assert r1["ROUTINE_YEARS"].equals(r2["ROUTINE_YEARS"])


# ---------------------------------------------------------------------------
# 3. create_signals()
# ---------------------------------------------------------------------------

class TestCreateSignals:
    def _make_classified_df(self, rows):
        """Build a classified DataFrame from (cik, date, type, code, shares, price) tuples."""
        data = []
        for issuer, date_str, trade_type, code, shares, price in rows:
            data.append({
                "ISSUERCIK": issuer,
                "TRANS_DATE": pd.Timestamp(date_str),
                "TRADE_TYPE": trade_type,
                "TRANSACTION_CODE": code,
                "TRANS_SHARES": shares,
                "TRANS_PRICEPERSHARE": price,
            })
        return pd.DataFrame(data)

    def test_signal_columns(self):
        """Output must have OPP_BUY, OPP_SELL, ROUT_BUY, ROUT_SELL, OPP_NET, ROUTINE_NET."""
        df = self._make_classified_df([
            ("12345", "2024-01-15", "OPPORTUNISTIC", "P", 100, 50.0),
            ("12345", "2024-01-20", "OPPORTUNISTIC", "S", 50, 52.0),
            ("12345", "2024-02-10", "ROUTINE", "P", 200, 48.0),
        ])
        signals = ia.create_signals(df)
        for col in ["OPPO_BUY", "OPPO_SELL", "ROUT_BUY", "ROUT_SELL",
                     "OPP_NET", "ROUTINE_NET"]:
            assert col in signals.columns, f"Missing column: {col}"

    def test_net_calculation(self):
        """OPP_NET = OPPO_BUY − OPPO_SELL (by count)."""
        df = self._make_classified_df([
            ("12345", "2024-01-15", "OPPORTUNISTIC", "P", 100, 50.0),
            ("12345", "2024-01-16", "OPPORTUNISTIC", "P", 200, 51.0),
            ("12345", "2024-01-20", "OPPORTUNISTIC", "S", 50, 52.0),
        ])
        signals = ia.create_signals(df, aggregate_by="count")
        row = signals[signals["ISSUERCIK"] == "12345"]
        assert row["OPPO_BUY"].iloc[0] == 2  # 2 buy trades
        assert row["OPPO_SELL"].iloc[0] == 1  # 1 sell trade
        assert row["OPP_NET"].iloc[0] == 1    # 2 − 1

    def test_aggregate_by_shares(self):
        """aggregate_by='shares' should sum TRANS_SHARES."""
        df = self._make_classified_df([
            ("12345", "2024-01-15", "OPPORTUNISTIC", "P", 100, 50.0),
            ("12345", "2024-01-16", "OPPORTUNISTIC", "P", 300, 51.0),
        ])
        signals = ia.create_signals(df, aggregate_by="shares")
        row = signals[signals["ISSUERCIK"] == "12345"]
        assert row["OPPO_BUY"].iloc[0] == 400.0

    def test_aggregate_by_value(self):
        """aggregate_by='value' should sum shares × price."""
        df = self._make_classified_df([
            ("12345", "2024-01-15", "OPPORTUNISTIC", "P", 100, 50.0),
        ])
        signals = ia.create_signals(df, aggregate_by="value")
        row = signals[signals["ISSUERCIK"] == "12345"]
        assert row["OPPO_BUY"].iloc[0] == 5000.0


# ---------------------------------------------------------------------------
# 4. Summary statistics
# ---------------------------------------------------------------------------

class TestSummaryStats:
    def test_round_trip(self):
        """After classification, routine + opportunistic = total."""
        rows = []
        for i in range(10):
            rows.append(("AAA", f"2021-01-{10+i:02d}", "P", 100))
            rows.append(("AAA", f"2022-01-{10+i:02d}", "P", 100))
            rows.append(("AAA", f"2023-01-{10+i:02d}", "P", 100))
            # 3 years → routine for January
        df = pd.DataFrame(rows, columns=["RPTOWNERCIK", "TRANS_DATE",
                                          "TRANSACTION_CODE", "TRANS_SHARES"])
        df["TRANS_DATE"] = pd.to_datetime(df["TRANS_DATE"])
        df["ISSUERCIK"] = "12345"
        classified = ia.classify_insiders(df)
        stats = ia.get_summary_stats(classified)
        assert stats["total_trades"] == 30
        assert stats["routine_trades"] + stats["opportunistic_trades"] == 30


# ---------------------------------------------------------------------------
# 5. InsiderTradingAnalyzer convenience class
# ---------------------------------------------------------------------------

class TestInsiderTradingAnalyzer:
    def test_convenience_workflow(self, tmp_path):
        """Full workflow through the analyzer class."""
        acc = "000001-24-001"
        _make_zip(
            tmp_path / "insider_2024q1.zip",
            trans_rows=[
                {"ACCESSION_NUMBER": acc, "TRANS_DATE": "15-JAN-2024",
                 "TRANSACTION_CODE": "P", "TRANS_SHARES": "1000",
                 "TRANS_PRICEPERSHARE": "50.00", "TRANS_ACQUIRED_DISP_CD": "A"},
            ],
            sub_rows=[
                {"ACCESSION_NUMBER": acc, "ISSUERCIK": "12345",
                 "ISSUERTRADINGSYMBOL": "ABC", "PERIODOFREPORT": ""},
            ],
            owner_rows=[
                {"ACCESSION_NUMBER": acc, "RPTOWNERCIK": "99999",
                 "RPTOWNERNAME": "JOHN DOE", "RPTOWNERRELATIONSHIP": "CEO",
                 "RPTOWNERTITLE": ""},
            ],
        )

        analyzer = ia.InsiderTradingAnalyzer(data_dir=str(tmp_path))
        trans = analyzer.load_all_quarters(start_year=2024, end_year=2024)
        assert len(trans) == 1

        classified = analyzer.classify_insiders()
        assert "TRADE_TYPE" in classified.columns

        signals = analyzer.create_signals()
        assert "OPP_NET" in signals.columns

        results = analyzer.run_analysis()
        assert "summary_stats" in results
        assert results["summary_stats"]["total_trades"] == 1
