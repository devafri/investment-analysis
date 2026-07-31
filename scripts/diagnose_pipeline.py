#!/usr/bin/env python3
"""
diagnose_pipeline.py
---------------------
Run this directly against your actual SEC zip/data folder to see exactly
where the funnel goes to zero, instead of guessing. Prints row counts at
every stage of the pipeline plus a per-threshold breakdown of the quality
screen.

USAGE
    python3 diagnose_pipeline.py --data-dir /path/to/2024q1.zip --form 10-K
    python3 diagnose_pipeline.py --data-dir /path/to/unzipped_folder --form 10-K
"""

import argparse
import sys
import tempfile
import zipfile
from pathlib import Path

import duckdb
import core.sec_value_screen as sec_screen


def resolve_dir(data_dir: str) -> Path:
    p = Path(data_dir).expanduser()
    if p.is_file() and p.suffix.lower() == ".zip":
        tmp = Path(tempfile.mkdtemp())
        with zipfile.ZipFile(p) as z:
            z.extractall(tmp)
        return tmp
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--form", default="10-K")
    args = ap.parse_args()

    data_path = resolve_dir(args.data_dir)
    for required in ("sub.txt", "num.txt"):
        if not (data_path / required).exists():
            sys.exit(f"ERROR: {required} not found in {data_path}")

    con = duckdb.connect()
    print(f"[1] Loading raw files from {data_path} ...")
    sec_screen.load_data(con, str(data_path))

    raw_sub = con.execute("SELECT COUNT(*) FROM sub").fetchone()[0]
    raw_num = con.execute("SELECT COUNT(*) FROM num").fetchone()[0]
    print(f"    sub.txt rows: {raw_sub:,}   num.txt rows: {raw_num:,}")

    forms = con.execute(
        "SELECT form, COUNT(*) AS n FROM sub GROUP BY form ORDER BY n DESC LIMIT 15"
    ).fetchdf()
    print(f"\n    Distinct form values actually present in sub.txt (top 15):")
    print(forms.to_string(index=False))

    print(f"\n[2] Filtering to form='{args.form}' ...")
    sec_screen.filter_submissions(con, args.form)
    matched = con.execute("SELECT COUNT(*) FROM sub_filtered").fetchone()[0]
    print(f"    sub_filtered rows: {matched:,}")
    if matched == 0:
        print(f"    >>> STOP: no filings matched '{args.form}'. Check the exact")
        print(f"        strings above -- this is very likely your issue if the")
        print(f"        UI is sending something that doesn't exactly match.")
        return

    annual = args.form.strip().upper() == "10-K"
    print(f"\n[3] Building fundamentals (annual={annual}) ...")
    fundamentals = sec_screen.build_fundamentals(con, annual=annual)
    print(f"    fundamentals rows: {len(fundamentals):,}")
    if fundamentals.empty:
        print("    >>> STOP: sub_filtered matched rows but no adsh values overlapped")
        print("        with num.txt. Confirm sub.txt and num.txt are from the SAME")
        print("        quarterly zip (not mismatched from two different downloads).")
        return

    print(f"\n[4] Tag coverage (non-null count per fundamental field, out of {len(fundamentals)}):")
    tag_cols = [c for c in fundamentals.columns if c not in
                ("adsh", "cik", "name", "sic", "period", "fy", "form", "filed")]
    coverage = fundamentals[tag_cols].notna().sum().sort_values(ascending=False)
    print(coverage.to_string())

    print(f"\n[5] Computing ratios ...")
    ratios = sec_screen.compute_ratios(fundamentals)
    print(f"    non-null ROIC: {ratios['ROIC'].notna().sum():,} / {len(ratios):,}")
    print(f"    non-null OperatingMargin: {ratios['OperatingMargin'].notna().sum():,}")
    print(f"    non-null DebtToEquity: {ratios['DebtToEquity'].notna().sum():,}")
    print(f"    non-null InterestCoverage: {ratios['InterestCoverage'].notna().sum():,}")
    print(f"    non-null CFO_to_NI: {ratios['CFO_to_NI'].notna().sum():,}")

    print(f"\n[6] Per-threshold pass counts (isolated, at default thresholds):")
    print(f"    ROIC > 0.15:              {(ratios['ROIC'] > 0.15).sum():,}")
    print(f"    OperatingMargin > 0.10:   {(ratios['OperatingMargin'] > 0.10).sum():,}")
    print(f"    DebtToEquity < 1.0:       {(ratios['DebtToEquity'] < 1.0).sum():,}")
    print(f"    InterestCoverage > 5:     {(ratios['InterestCoverage'] > 5).sum():,}")
    print(f"    CFO_to_NI > 0.8:          {(ratios['CFO_to_NI'] > 0.8).sum():,}")
    print(f"    NetIncomeLoss > 0:        {(ratios['NetIncomeLoss'] > 0).sum():,}")

    screened = sec_screen.apply_quality_screen(ratios)
    print(f"\n[7] Combined screen (all thresholds together): {len(screened):,} companies pass")
    if not screened.empty:
        print(screened[["name", "ROIC", "OperatingMargin", "DebtToEquity", "InterestCoverage"]].head(20).to_string(index=False))


if __name__ == "__main__":
    main()