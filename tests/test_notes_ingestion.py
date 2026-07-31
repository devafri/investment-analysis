"""Tests for core/notes_ingestion.py and qualitative red-flag scanning."""

import duckdb
import pandas as pd
import pytest

from core.fundamentals.company_analysis import scan_footnotes_for_red_flags, _strip_html
from core.fundamentals.notes_ingestion import ingest_notes_txt
from tests.conftest import write_sub_txt


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_txt_txt(path, rows):
    """Write a minimal txt.txt file for testing."""
    import io
    columns = ["adsh", "tag", "version", "ddate", "qtrs", "uom", "value", "footnote"]
    with open(path, "w") as f:
        f.write("\t".join(columns) + "\n")
        for row in rows:
            f.write("\t".join(str(row.get(c, "")) for c in columns) + "\n")


# ---------------------------------------------------------------------------
# 1. Ingestion
# ---------------------------------------------------------------------------

class TestNotesIngestion:
    def test_ingest_txt_creates_table(self, tmp_path):
        """Ingesting a txt.txt should create footnote_text_blocks."""
        adsh = "0000012345-24-000001"
        write_sub_txt(tmp_path / "sub.txt", [
            {"adsh": adsh, "cik": "12345", "name": "TEST CORP", "sic": "7370",
             "form": "10-K", "period": "20231231", "fy": "2023", "fp": "FY",
             "filed": "20240215"},
        ])
        _write_txt_txt(tmp_path / "txt.txt", [
            {"adsh": adsh, "tag": "SubstantialDoubtAboutGoingConcernTextBlock",
             "version": "us-gaap/2023", "ddate": "20231231", "qtrs": "0",
             "uom": "", "value": "The Company has substantial doubt about its ability to continue as a going concern..."},
            {"adsh": adsh, "tag": "DebtDisclosureTextBlock",
             "version": "us-gaap/2023", "ddate": "20231231", "qtrs": "0",
             "uom": "", "value": "The Company's total debt outstanding is $500M as of December 31, 2023."},
            # This tag is NOT in HIGH_VALUE_TAGS — should be skipped
            {"adsh": adsh, "tag": "SomeRandomDisclosure",
             "version": "us-gaap/2023", "ddate": "20231231", "qtrs": "0",
             "uom": "", "value": "Random text that should be ignored."},
        ])

        con = duckdb.connect()
        # Load sub.txt first — in the real pipeline this is done by
        # load_data_filtered() before ingest_notes_txt() is called.
        con.execute(f"""
            CREATE TABLE sub AS
            SELECT * FROM read_csv('{tmp_path / "sub.txt"}',
                delim='\t', header=True, quote='', all_varchar=True,
                strict_mode=False);
        """)
        count = ingest_notes_txt(con, str(tmp_path))
        con.close()

        assert count == 2, (
            f"Expected 2 high-value text blocks, got {count}"
        )

    def test_no_txt_file_returns_zero(self, tmp_path):
        """When txt.txt is absent, ingest_notes_txt should return 0 silently."""
        con = duckdb.connect()
        count = ingest_notes_txt(con, str(tmp_path))
        con.close()
        assert count == 0

    def test_txt_with_no_high_value_tags(self, tmp_path):
        """Only high-value tags should be ingested."""
        adsh = "0000012345-24-000001"
        write_sub_txt(tmp_path / "sub.txt", [
            {"adsh": adsh, "cik": "12345", "name": "TEST CORP", "sic": "7370",
             "form": "10-K", "period": "20231231", "fy": "2023", "fp": "FY",
             "filed": "20240215"},
        ])
        _write_txt_txt(tmp_path / "txt.txt", [
            {"adsh": adsh, "tag": "SomeRandomDisclosure",
             "version": "us-gaap/2023", "ddate": "20231231", "qtrs": "0",
             "uom": "", "value": "Random text."},
        ])

        con = duckdb.connect()
        con.execute(f"""
            CREATE TABLE sub AS
            SELECT * FROM read_csv('{tmp_path / "sub.txt"}',
                delim='\t', header=True, quote='', all_varchar=True,
                strict_mode=False);
        """)
        count = ingest_notes_txt(con, str(tmp_path))
        con.close()
        assert count == 0


# ---------------------------------------------------------------------------
# 2. HTML stripping
# ---------------------------------------------------------------------------

class TestHtmlStripping:
    def test_strips_tags(self):
        html = "<p>The Company <b>may not</b> continue as a going concern.</p>"
        assert "going concern" in _strip_html(html)
        assert "<p>" not in _strip_html(html)
        assert "<b>" not in _strip_html(html)

    def test_decodes_entities(self):
        html = "Assets &amp; Liabilities are material."
        result = _strip_html(html)
        assert "&" in result
        assert "&amp;" not in result

    def test_collapses_whitespace(self):
        html = "First   sentence.\n\nSecond  paragraph."
        result = _strip_html(html)
        assert "  " not in result
        assert "First sentence." in result


# ---------------------------------------------------------------------------
# 3. Qualitative red-flag scanning
# ---------------------------------------------------------------------------

class TestQualitativeRedFlags:
    def test_detects_going_concern(self):
        df = pd.DataFrame([{
            "tag": "SubstantialDoubtAboutGoingConcernTextBlock",
            "txt_value": "<p>There is substantial doubt about the Company's "
                         "ability to continue as a <b>going concern</b> within "
                         "the next twelve months.</p>",
        }])
        flags = scan_footnotes_for_red_flags(df)
        assert len(flags) == 1
        assert flags[0]["keyword"] == "Going concern"
        assert "going concern" in flags[0]["snippet"].lower()

    def test_detects_material_weakness(self):
        df = pd.DataFrame([{
            "tag": "BasisOfPresentationAndSignificantAccountingPoliciesTextBlock",
            "txt_value": "Management identified a material weakness in internal "
                         "control over financial reporting related to segregation "
                         "of duties.",
        }])
        flags = scan_footnotes_for_red_flags(df)
        assert len(flags) >= 1
        keywords = [f["keyword"] for f in flags]
        assert "Material weakness" in keywords

    def test_detects_multiple_keywords(self):
        """One text block can match multiple keyword patterns."""
        df = pd.DataFrame([{
            "tag": "LegalMattersAndContingenciesTextBlock",
            "txt_value": "The Company is subject to an SEC investigation "
                         "regarding its revenue recognition practices. The "
                         "Company also restated its prior period financial "
                         "statements due to accounting errors.",
        }])
        flags = scan_footnotes_for_red_flags(df)
        keywords = {f["keyword"] for f in flags}
        assert "Regulatory investigation" in keywords
        assert "Restatement" in keywords

    def test_empty_dataframe(self):
        flags = scan_footnotes_for_red_flags(pd.DataFrame())
        assert flags == []

    def test_no_keywords_returns_empty(self):
        df = pd.DataFrame([{
            "tag": "DebtDisclosureTextBlock",
            "txt_value": "The Company maintains a revolving credit facility "
                         "with a syndicate of banks. Interest is calculated at "
                         "SOFR plus 150 basis points.",
        }])
        flags = scan_footnotes_for_red_flags(df)
        assert flags == []

    def test_snippet_truncation(self):
        """Snippets should be reasonably sized."""
        df = pd.DataFrame([{
            "tag": "SubstantialDoubtAboutGoingConcernTextBlock",
            "txt_value": "x" * 50 + " going concern " + "y" * 500,
        }])
        flags = scan_footnotes_for_red_flags(df)
        assert len(flags) == 1
        assert len(flags[0]["snippet"]) <= 300  # should be ~280 max
        assert "going concern" in flags[0]["snippet"]
