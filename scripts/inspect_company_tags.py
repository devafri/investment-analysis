#!/usr/bin/env python3
"""
inspect_company_tags.py
-------------------------
Diagnostic: shows exactly what raw XBRL tags a specific company reported in
your real SEC data dump, so you can see WHY a mapped field (e.g. TotalDebt)
came up null instead of guessing. sec_value_screen.py's TAG_MAP only pulls
values for a fixed list of known tag names (e.g. LongTermDebtNoncurrent) --
if a filer uses a different or custom tag for the same concept, it will
silently show up as null with no error, which is almost certainly what's
happening here.

USAGE
    # Find by company name (partial match, case-insensitive)
    python3 inspect_company_tags.py --data-dir ./data/2024q1 --name "CSX"

    # Find by exact CIK
    python3 inspect_company_tags.py --data-dir ./data/2024q1 --cik 277948

    # Narrow to tags matching a keyword (default: debt)
    python3 inspect_company_tags.py --data-dir ./data/2024q1 --name "CSX" --keyword debt

WHAT IT SHOWS
- The matching submission(s) (adsh, form, period, filed)
- Every row in num.txt for that filing where the tag name contains your
  keyword (case-insensitive) -- tag, qtrs, ddate, coreg, value
- Whether the tag is a standard US-GAAP tag or a company-specific custom
  extension (via tag.txt's `custom` flag) -- custom tags won't match
  sec_value_screen.py's TAG_MAP at all unless you add them explicitly
- If multiple rows exist for the same tag (e.g. different `coreg` values for
  subsidiary/co-registrant breakdowns), all of them are shown so you can see
  whether the pivot's MAX()-based aggregation is at risk of picking the wrong
  one for a company with a complex debt structure
"""

import argparse
import sys
import zipfile
import tempfile
from pathlib import Path

import duckdb


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
    ap.add_argument("--name", help="Company name substring (case-insensitive)")
    ap.add_argument("--cik", help="Exact CIK")
    ap.add_argument("--keyword", default="debt", help="Only show tags containing this substring (case-insensitive). Use '' to show ALL tags for the filing.")
    args = ap.parse_args()

    if not args.name and not args.cik:
        sys.exit("ERROR: provide --name or --cik")

    data_path = resolve_dir(args.data_dir)
    for required in ("sub.txt", "num.txt", "tag.txt"):
        if not (data_path / required).exists():
            sys.exit(f"ERROR: {required} not found in {data_path}")

    con = duckdb.connect()
    con.execute(f"""
        CREATE TABLE sub AS SELECT * FROM read_csv('{data_path}/sub.txt',
            delim='\t', header=True, quote='', all_varchar=True, strict_mode=False);
    """)
    con.execute(f"""
        CREATE TABLE num AS SELECT * FROM read_csv('{data_path}/num.txt',
            delim='\t', header=True, quote='', all_varchar=True, strict_mode=False);
    """)
    con.execute(f"""
        CREATE TABLE tag AS SELECT * FROM read_csv('{data_path}/tag.txt',
            delim='\t', header=True, quote='', all_varchar=True, strict_mode=False);
    """)

    if args.cik:
        where = f"cik = '{args.cik}'"
    else:
        safe_name = args.name.replace("'", "''").upper()
        where = f"UPPER(name) LIKE '%{safe_name}%'"

    filings = con.execute(f"""
        SELECT adsh, cik, name, form, period, fy, filed
        FROM sub WHERE {where}
        ORDER BY filed DESC
    """).fetchdf()

    if filings.empty:
        print("No matching company found in sub.txt. Try a broader --name substring or double check --cik.")
        return

    print(f"Found {len(filings)} matching filing(s):")
    print(filings.to_string(index=False))

    for _, filing in filings.iterrows():
        adsh = filing["adsh"]
        print(f"\n{'='*100}\nFiling {adsh}  ({filing['form']}, period {filing['period']}, filed {filing['filed']})")

        kw = args.keyword.upper()
        rows = con.execute(f"""
            SELECT n.tag, n.qtrs, n.ddate, n.coreg, n.value, n.uom,
                   t.custom, t.tlabel
            FROM num n
            LEFT JOIN tag t ON n.tag = t.tag AND n.version = t.version
            WHERE n.adsh = '{adsh}'
              AND UPPER(n.tag) LIKE '%{kw}%'
            ORDER BY n.tag, n.qtrs, n.coreg
        """).fetchdf()

        if rows.empty:
            print(f"  No tags containing '{args.keyword}' found for this filing.")
            print(f"  Try --keyword '' to dump ALL tags and manually scan for the debt line item.")
            continue

        print(rows.to_string(index=False))

        custom_tags = rows[rows["custom"] == "1"]["tag"].unique() if "custom" in rows.columns else []
        if len(custom_tags):
            print(f"\n  NOTE: these are CUSTOM (company-specific) extension tags, not standard "
                  f"US-GAAP: {list(custom_tags)}")
            print(f"  sec_value_screen.py's TAG_MAP only matches standard tag names -- a custom "
                  f"tag here means the value will show as null unless you add it to TAG_MAP explicitly.")

        dup_check = rows.groupby(["tag", "qtrs"]).size()
        dupes = dup_check[dup_check > 1]
        if len(dupes):
            print(f"\n  NOTE: multiple rows exist for the same tag+qtrs combination (likely different "
                  f"`coreg` values for subsidiary/co-registrant breakdowns). The current pivot logic "
                  f"takes MAX(value) across these, which may not be the consolidated total you want:")
            print(dupes.to_string())


if __name__ == "__main__":
    main()