"""Company-level analysis: red flags, trend tables, and forensic checks.

Pure functions — no FastAPI, no DuckDB. Extracted from app.py so they can be
tested directly without going through HTTP.
"""

import re
from typing import Any, Dict, List, Optional, TypedDict

import pandas as pd

import core.formatting as formatting


# ---------------------------------------------------------------------------
# Qualitative red-flag keyword scanning (footnote text blocks)
# ---------------------------------------------------------------------------

class QualitativeFlag(TypedDict):
    keyword: str
    tag: str
    tag_label: str
    snippet: str  # first ~250 chars of the matching text


# (keyword pattern, display label)
_KEYWORD_PATTERNS: List[tuple] = [
    (r"going\s+concern", "Going concern"),
    (r"material\s+weakness", "Material weakness"),
    (r"restatement|restated|restating", "Restatement"),
    (r"default\s+(on\s+|under\s+)?covenant", "Default on covenant"),
    (r"SEC\s+investigation|DOJ\s+investigation|regulatory\s+investigation",
     "Regulatory investigation"),
    (r"class\s+action|shareholder\s+litigation|securities\s+litigation",
     "Shareholder litigation"),
    (r"internal\s+control\s+over\s+financial\s+reporting.*?(ineffective|not\s+effective)",
     "Ineffective internal controls"),
    (r"restate(d|ment)\s+(prior|previously|financial)",
     "Prior period restatement"),
]


def scan_footnotes_for_red_flags(
    footnotes_df: pd.DataFrame,
) -> List[QualitativeFlag]:
    """Scan footnote text blocks for qualitative red-flag keywords.

    Parameters
    ----------
    footnotes_df : pd.DataFrame
        Must have columns ``tag``, ``txt_value``.  Typically comes from
        :func:`core.notes_ingestion.get_footnotes_for_cik`.

    Returns
    -------
    list[QualitativeFlag]
        One entry per matched keyword per footnote block.  If the same
        keyword appears in multiple blocks, each gets its own entry (the
        UI can deduplicate by keyword if desired).
    """
    if footnotes_df.empty:
        return []

    results: List[QualitativeFlag] = []

    for _, row in footnotes_df.iterrows():
        text = row.get("txt_value") or ""
        if not text:
            continue

        # Strip HTML tags so keyword matching works on the plain text
        # (SEC footnote blocks often contain embedded HTML).
        plain = _strip_html(text)

        tag = row.get("tag") or ""
        tag_label = formatting.FOOTNOTE_TAG_LABELS.get(tag, tag)

        for pattern, keyword_label in _KEYWORD_PATTERNS:
            match = re.search(pattern, plain, re.IGNORECASE)
            if match:
                # Snippet — start a few words before the match for context
                start = max(0, match.start() - 40)
                end = min(len(plain), match.end() + 210)
                snippet = plain[start:end].strip()
                if start > 0:
                    snippet = "…" + snippet
                if end < len(plain):
                    snippet = snippet + "…"

                results.append({
                    "keyword": keyword_label,
                    "tag": tag,
                    "tag_label": tag_label,
                    "snippet": snippet[:280],
                })

    return results


def _strip_html(text: str) -> str:
    """Remove HTML tags and decode common entities so keyword scanning
    works on the plain text rather than raw HTML markup."""
    # Remove HTML tags
    plain = re.sub(r"<[^>]+>", " ", text)
    # Decode common entities
    plain = plain.replace("&amp;", "&")
    plain = plain.replace("&lt;", "<")
    plain = plain.replace("&gt;", ">")
    plain = plain.replace("&quot;", '"')
    plain = plain.replace("&#39;", "'")
    plain = plain.replace("&nbsp;", " ")
    plain = plain.replace("&#160;", " ")
    # Collapse whitespace
    plain = re.sub(r"\s+", " ", plain)
    return plain.strip()


def compute_red_flags(history_df: pd.DataFrame) -> List[str]:
    """Analyze filing history and return a list of red-flag strings.

    Only checks 10-K (annual) filings to avoid false positives from mixing
    annual and quarterly figures (e.g. Q1 vs full-year magnitudes).
    """
    if history_df.empty or len(history_df) < 3:
        return []

    flags: List[str] = []
    recent = history_df.head(min(8, len(history_df)))
    ni_vals = [r.get("NetIncomeLoss") for _, r in recent.iterrows()
               if r.get("NetIncomeLoss") is not None]
    rev_vals = [r.get("Revenues") for _, r in recent.iterrows()
                if r.get("Revenues") is not None]
    de_vals = [r.get("DebtToEquity") for _, r in recent.iterrows()
               if r.get("DebtToEquity") is not None]
    fcf_vals = [r.get("FCF") for _, r in recent.iterrows()
                if r.get("FCF") is not None]

    # Days Sales Outstanding (DSO) channel-stuffing check.
    # DSO = (AccountsReceivable / Revenue) * 365.
    # If DSO ballooned while revenue grew, receivables are growing faster than
    # sales — a possible channel-stuffing or collection-problem signal.
    ar_rows = [(r.get("AccountsReceivable"), r.get("Revenues"))
               for _, r in recent.iterrows()
               if r.get("AccountsReceivable") is not None
               and r.get("Revenues") is not None
               and float(r.get("Revenues") or 0) > 0]
    if len(ar_rows) >= 2:
        ar_cur, rev_cur = ar_rows[0]
        ar_pri, rev_pri = ar_rows[1]
        dso_cur = (float(ar_cur) / float(rev_cur)) * 365
        dso_pri = (float(ar_pri) / float(rev_pri)) * 365
        dso_change = (dso_cur / dso_pri) - 1 if dso_pri > 0 else 0
        rev_grew = float(rev_cur) > float(rev_pri)
        if dso_change > 0.20 and rev_grew:
            flags.append(
                f"DSO increased {dso_change * 100:.0f}% while revenue grew "
                f"({dso_pri:.0f} → {dso_cur:.0f} days) — possible "
                f"channel-stuffing or deteriorating collections"
            )


    if len(rev_vals) >= 3:
        declines = sum(1 for i in range(1, min(len(rev_vals), 5))
                       if float(rev_vals[i]) < float(rev_vals[i - 1]))
        if declines >= 2:
            flags.append(
                f"Revenue declined in {declines} of the last "
                f"{min(len(rev_vals) - 1, 4)} periods"
            )
    # Net income negative in most recent period
    if ni_vals and float(ni_vals[0]) < 0:
        flags.append("Net income negative in most recent filing")
    # FCF negative recently
    if fcf_vals and float(fcf_vals[0]) < 0:
        flags.append("Free cash flow negative in most recent filing")
    # Rising leverage over time
    if len(de_vals) >= 3:
        if float(de_vals[0]) > float(de_vals[-1]) * 1.5:
            flags.append(
                "Debt/Equity has increased >50% across available filings"
            )
    return flags


def build_trend_table(
    history_df: pd.DataFrame, metrics: List[str]
) -> List[Dict[str, Any]]:
    """Build a list-of-dicts trend table showing the last 6 periods for key
    metrics.

    Uses the key ``metrics`` (NOT ``values``) because Jinja2 resolves
    ``row.values`` as the Python dict method, not the item lookup.
    """
    if history_df.empty:
        return []
    recent = history_df.head(6)
    rows: List[Dict[str, Any]] = []
    for _, row_data in recent.iterrows():
        rows.append({
            "period": formatting.format_display_value(
                "period", row_data.get("period")
            ),
            "form": row_data.get("form", ""),
            "fp": row_data.get("fp", ""),
            "metrics": {
                m: formatting.format_display_value(m, row_data.get(m))
                for m in metrics
            },
        })
    return rows
