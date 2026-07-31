"""Display formatting: how to render raw fundamental/ratio values (which are
unlabeled floats like 1515906000.0, 0.15032662185839576, 20251231) as
readable text ("$1.52B", "15.03%", "Dec 31, 2025"), and how to group fields
on the company detail page. Pure functions -- no FastAPI, no DuckDB.
"""

import math
from datetime import datetime
from typing import Any, Dict

# How to display each fundamental/ratio field.
FIELD_FORMAT: Dict[str, str] = {
    # currency (dollar) fields
    "Revenues": "usd", "CostOfRevenue": "usd", "GrossProfit": "usd",
    "OperatingIncomeLoss": "usd", "NetIncomeLoss": "usd", "InterestExpense": "usd",
    "IncomeTaxExpenseBenefit": "usd", "NetCashProvidedByUsedInOperatingActivities": "usd",
    "CapitalExpenditures": "usd", "Assets": "usd", "AssetsCurrent": "usd",
    "LiabilitiesCurrent": "usd", "CashAndCashEquivalents": "usd",
    "PropertyPlantAndEquipmentNet": "usd", "StockholdersEquity": "usd",
    "LongTermDebtNoncurrent": "usd", "LongTermDebtCurrent": "usd",
    "ShortTermBorrowings": "usd", "OperatingLeaseLiability": "usd",
    "TotalDebt": "usd", "NetDebt": "usd",
    "NWC": "usd", "InvestedCapital": "usd", "FCF": "usd",
    "EnterpriseValue": "usd",
    # percentages (stored as decimal fractions, e.g. 0.15 -> "15.00%")
    "ROIC": "pct", "GrossMargin": "pct", "OperatingMargin": "pct", "NetMargin": "pct",
    "ROE": "pct", "ROA": "pct", "AccrualRatio": "pct", "EarningsYield": "pct",
    "RevenueGrowth": "pct", "RevenueGrowth3yr": "pct", "FCFGrowth": "pct",
    "MarginOfSafetyGraham": "pct", "MarginOfSafetyDCF": "pct",
    # multiples ("x" suffix)
    "CurrentRatio": "multiple", "DebtToEquity": "multiple",
    "InterestCoverage": "multiple", "CFO_to_NI": "multiple",
    "PE": "multiple", "PB": "multiple", "EVToEBIT": "multiple", "PFCF": "multiple",
    # dates (stored as YYYYMMDD)
    "period": "date", "filed": "date", "as_of_period": "date",
    # plain counts
    "CommonStockSharesOutstanding": "count",
    "CommonStockSharesOutstandingCoverPage": "count",
    "DepreciationAndAmortization": "usd",
    "Goodwill": "usd",
    "AccountsReceivable": "usd",
    "IntangibleAssetsNet": "usd",
    "InterestIncome": "usd",
    # plain integer, no comma/decimal formatting
    "fy": "year",
}

# Which section of the company detail page each field belongs in.
GROUP_MAP: Dict[str, str] = {
    "Revenues": "Profitability", "CostOfRevenue": "Profitability", "GrossProfit": "Profitability",
    "GrossMargin": "Profitability", "OperatingIncomeLoss": "Profitability", "OperatingMargin": "Profitability",
    "NetIncomeLoss": "Profitability", "NetMargin": "Profitability", "IncomeTaxExpenseBenefit": "Profitability",
    "ROIC": "Profitability", "ROE": "Profitability", "ROA": "Profitability",

    "InterestExpense": "Leverage & Solvency", "Assets": "Leverage & Solvency", "AssetsCurrent": "Leverage & Solvency",
    "LiabilitiesCurrent": "Leverage & Solvency", "StockholdersEquity": "Leverage & Solvency",
    "LongTermDebtNoncurrent": "Leverage & Solvency", "LongTermDebtCurrent": "Leverage & Solvency",
    "ShortTermBorrowings": "Leverage & Solvency", "OperatingLeaseLiability": "Leverage & Solvency",
    "TotalDebt": "Leverage & Solvency", "NetDebt": "Leverage & Solvency",
    "CurrentRatio": "Leverage & Solvency", "DebtToEquity": "Leverage & Solvency", "InterestCoverage": "Leverage & Solvency",
    "CashAndCashEquivalents": "Leverage & Solvency", "AccountsReceivable": "Leverage & Solvency",
    "PropertyPlantAndEquipmentNet": "Leverage & Solvency",

    "NetCashProvidedByUsedInOperatingActivities": "Earnings Quality", "CFO_to_NI": "Earnings Quality",
    "AccrualRatio": "Earnings Quality", "FCF": "Earnings Quality", "CapitalExpenditures": "Earnings Quality",

    "NWC": "Other Fundamentals", "InvestedCapital": "Other Fundamentals",
    "CommonStockSharesOutstanding": "Other Fundamentals",
    "CommonStockSharesOutstandingCoverPage": "Other Fundamentals",
    "DepreciationAndAmortization": "Other Fundamentals",
    "Goodwill": "Other Fundamentals", "IntangibleAssetsNet": "Other Fundamentals",
    "InterestIncome": "Other Fundamentals",

    "period": "Filing Info", "fy": "Filing Info", "form": "Filing Info", "filed": "Filing Info", "adsh": "Filing Info",
    "as_of_period": "Filing Info", "ttm_basis": "Filing Info",

    "EnterpriseValue": "Market Data", "EarningsYield": "Market Data",
    "Ticker": "Market Data", "MagicFormulaRank": "Market Data",
    "Price": "Market Data", "MarketCap": "Market Data", "PE": "Market Data",
    "PB": "Market Data", "EVToEBIT": "Market Data", "PFCF": "Market Data",
    "MarginOfSafetyGraham": "Market Data", "MarginOfSafetyDCF": "Market Data",
    "High52Week": "Market Data", "Low52Week": "Market Data",

    # Growth metrics
    "RevenueGrowth": "Growth", "RevenueGrowth3yr": "Growth", "FCFGrowth": "Growth",

    # Valuation
    "GrahamNumber": "Valuation", "DCFIntrinsicValue": "Valuation",

    # F-Score
    "FScore": "F-Score",
}
GROUP_ORDER = ["Profitability", "Growth", "Leverage & Solvency", "Earnings Quality",
               "Valuation", "F-Score", "Market Data", "Other Fundamentals", "Filing Info"]


# ---------------------------------------------------------------------------
# SIC → Sector mapping (first 1-2 digits → broad industry sector)
# ---------------------------------------------------------------------------
SIC_TO_SECTOR = {
    "01": "Agriculture", "02": "Agriculture", "07": "Agriculture", "08": "Agriculture", "09": "Agriculture",
    "10": "Mining", "12": "Mining", "13": "Mining", "14": "Mining",
    "15": "Construction", "16": "Construction", "17": "Construction",
    "20": "Manufacturing", "21": "Manufacturing", "22": "Manufacturing", "23": "Manufacturing",
    "24": "Manufacturing", "25": "Manufacturing", "26": "Manufacturing", "27": "Manufacturing",
    "28": "Manufacturing", "29": "Manufacturing", "30": "Manufacturing", "31": "Manufacturing",
    "32": "Manufacturing", "33": "Manufacturing", "34": "Manufacturing", "35": "Manufacturing",
    "36": "Manufacturing", "37": "Manufacturing", "38": "Manufacturing", "39": "Manufacturing",
    "40": "Transportation", "41": "Transportation", "42": "Transportation", "44": "Transportation",
    "45": "Transportation", "46": "Transportation", "47": "Transportation", "48": "Communications",
    "49": "Utilities",
    "50": "Wholesale Trade", "51": "Wholesale Trade",
    "52": "Retail Trade", "53": "Retail Trade", "54": "Retail Trade", "55": "Retail Trade",
    "56": "Retail Trade", "57": "Retail Trade", "58": "Retail Trade", "59": "Retail Trade",
    "60": "Financial Services", "61": "Financial Services", "62": "Financial Services",
    "63": "Financial Services", "64": "Financial Services", "65": "Financial Services",
    "67": "Financial Services",
    "70": "Services", "72": "Services", "73": "Services", "75": "Services", "76": "Services",
    "78": "Services", "79": "Services", "80": "Services", "82": "Services", "83": "Services",
    "87": "Services", "88": "Services", "89": "Services",
    "91": "Public Administration", "92": "Public Administration", "93": "Public Administration",
    "94": "Public Administration", "95": "Public Administration", "96": "Public Administration",
    "97": "Public Administration", "99": "Nonclassifiable",
}


def sic_to_sector(sic) -> str:
    """Map a raw SIC code (int, float, or str) to a broad industry sector."""
    if sic is None:
        return "Unknown"
    s = str(int(float(sic))).zfill(4)
    # Try 2-digit prefix, then 1-digit
    for length in (2, 1):
        key = s[:length]
        if key in SIC_TO_SECTOR:
            return SIC_TO_SECTOR[key]
    return "Other"


def format_display_value(column: str, value: Any) -> str:
    """Render one fundamental/ratio value the way a human should read it,
    based on FIELD_FORMAT. Falls back to a plain comma-formatted number (or
    the raw string) for anything not explicitly classified."""
    if value is None:
        return "—"
    kind = FIELD_FORMAT.get(column, "number")

    if kind == "date":
        s = str(value).split(".")[0]
        if len(s) == 8 and s.isdigit():
            try:
                return datetime.strptime(s, "%Y%m%d").strftime("%b %d, %Y")
            except ValueError:
                return s
        return s

    try:
        v = float(value)
    except (TypeError, ValueError):
        return str(value)

    if math.isinf(v):
        return "∞ (no debt)" if column == "InterestCoverage" else "∞"
    if math.isnan(v):
        return "—"

    if kind == "pct":
        return f"{v * 100:.2f}%"
    if kind == "usd":
        sign = "-" if v < 0 else ""
        av = abs(v)
        if av >= 1_000_000_000:
            return f"{sign}${av / 1_000_000_000:.2f}B"
        if av >= 1_000_000:
            return f"{sign}${av / 1_000_000:.2f}M"
        if av >= 1_000:
            return f"{sign}${av / 1_000:.1f}K"
        return f"{sign}${av:,.2f}"
    if kind == "multiple":
        return f"{v:.2f}x"
    if kind == "count":
        return f"{v:,.0f}"
    if kind == "year":
        return f"{int(v)}"
    return f"{v:,.2f}"


# --- Jinja template filters (used directly in templates via |fmt_pct etc.) ---

def fmt_pct(value):
    if value is None:
        return "—"
    v = float(value)
    if math.isinf(v):
        return "∞"
    if math.isnan(v):
        return "—"
    return f"{v * 100:.2f}%"


def fmt_number(value):
    if value is None:
        return "—"
    v = float(value)
    if math.isinf(v):
        return "∞"
    if math.isnan(v):
        return "—"
    return f"{v:,.2f}"


def fmt_currency(value):
    if value is None:
        return "—"
    v = float(value)
    if math.isinf(v):
        return "∞"
    if math.isnan(v):
        return "—"
    sign = "-" if v < 0 else ""
    av = abs(v)
    if av >= 1_000_000_000:
        return f"{sign}${av / 1_000_000_000:.2f}B"
    if av >= 1_000_000:
        return f"{sign}${av / 1_000_000:.2f}M"
    if av >= 1_000:
        return f"{sign}${av / 1_000:.1f}K"
    return f"{sign}${av:,.2f}"


# ---------------------------------------------------------------------------
# Footnote tag → human-readable label mapping (used by notes ingestion UI)
# ---------------------------------------------------------------------------

FOOTNOTE_TAG_LABELS: Dict[str, str] = {
    "SubstantialDoubtAboutGoingConcernTextBlock": "Going Concern",
    "LiquidityVisAVisGoingConcernTextBlock": "Liquidity / Going Concern",
    "DebtDisclosureTextBlock": "Debt Disclosure",
    "CommitmentsAndContingenciesTextBlock": "Commitments & Contingencies",
    "LongTermDebtTextBlock": "Long-Term Debt",
    "ScheduleOfDebtTextBlock": "Debt Schedule",
    "RestatementToPriorPeriodBlock": "Restatement",
    "AccountingChangesAndErrorCorrectionsTextBlock": "Accounting Changes",
    "LegalMattersAndContingenciesTextBlock": "Legal Matters",
    "RelatedPartyTransactionsTextBlock": "Related Party Transactions",
    "BasisOfPresentationAndSignificantAccountingPoliciesTextBlock": "Accounting Policies",
    "RevenueFromContractWithCustomerTextBlock": "Revenue Recognition",
    "LesseeOperatingLeasesTextBlock": "Operating Leases",
}