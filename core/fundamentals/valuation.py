"""Valuation functions: Graham Number, DCF intrinsic value, and margin-of-safety
calculations. Pure functions — no DuckDB, no FastAPI. Vectorized where possible
to avoid the O(n) cost of df.iterrows() on large screening DataFrames.

Shared helpers (_safe_div) are defined once here and imported by sec_value_screen
so there is exactly one canonical implementation.
"""

import math
from typing import Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def safe_div(numerator, denominator) -> Optional[float]:
    """Divide two possibly-None/NaN values, returning None on any invalid
    input instead of raising.  Exported so sec_value_screen can import from
    a single canonical location."""
    if numerator is None or denominator is None:
        return None
    try:
        n, d = float(numerator), float(denominator)
    except (TypeError, ValueError):
        return None
    if math.isnan(n) or math.isnan(d) or d == 0:
        return None
    return n / d


# ---------------------------------------------------------------------------
# Graham Number
# ---------------------------------------------------------------------------

def compute_graham_number(
    eps: Optional[float],
    book_value_per_share: Optional[float],
) -> Optional[float]:
    """Benjamin Graham's defensive investor formula: √(22.5 × EPS × BVPS).

    Returns None for any invalid input (negative values, None, NaN) rather
    than producing a meaningless or complex result.
    """
    if eps is None or book_value_per_share is None:
        return None
    try:
        eps_f = float(eps)
        bv_f = float(book_value_per_share)
    except (TypeError, ValueError):
        return None
    if math.isnan(eps_f) or math.isnan(bv_f):
        return None
    if eps_f <= 0 or bv_f <= 0:
        return None
    return math.sqrt(22.5 * eps_f * bv_f)


# ---------------------------------------------------------------------------
# Discounted Cash Flow
# ---------------------------------------------------------------------------

def compute_dcf_intrinsic_value_per_share(
    fcf: Optional[float],
    shares_outstanding: Optional[float],
    growth_rate: float = 0.05,
    discount_rate: float = 0.10,
    terminal_growth_rate: float = 0.025,
    projection_years: int = 5,
) -> Optional[float]:
    """Two-stage DCF: project FCF at `growth_rate` for `projection_years`,
    then a perpetuity terminal value at `terminal_growth_rate`, discounted
    back at `discount_rate`. Divide by `shares_outstanding` for per-share
    intrinsic value.

    Returns None when inputs are invalid (zero/negative/NaN FCF, zero/NaN
    shares, or discount_rate <= terminal_growth_rate which would produce a
    nonsensical terminal value).
    """
    if fcf is None or shares_outstanding is None:
        return None
    try:
        fcf_f = float(fcf)
        shares_f = float(shares_outstanding)
    except (TypeError, ValueError):
        return None
    if math.isnan(fcf_f) or math.isnan(shares_f):
        return None
    if fcf_f <= 0 or shares_f <= 0:
        return None
    if discount_rate <= terminal_growth_rate:
        return None

    total_pv = 0.0
    projected_fcf = fcf_f
    for year in range(1, projection_years + 1):
        projected_fcf = fcf_f * ((1 + growth_rate) ** year)
        pv = projected_fcf / ((1 + discount_rate) ** year)
        total_pv += pv

    # Terminal value (perpetuity growth from end of projection period)
    terminal_fcf = projected_fcf * (1 + terminal_growth_rate)
    terminal_value = terminal_fcf / (discount_rate - terminal_growth_rate)
    pv_terminal = terminal_value / ((1 + discount_rate) ** projection_years)
    total_pv += pv_terminal

    return total_pv / shares_f


# ---------------------------------------------------------------------------
# Batch valuation — vectorized where possible
# ---------------------------------------------------------------------------

def apply_valuation(
    df: pd.DataFrame,
    growth_rate: float = 0.05,
    discount_rate: float = 0.10,
    terminal_growth_rate: float = 0.025,
    projection_years: int = 5,
) -> pd.DataFrame:
    """Adds GrahamNumber and DCFIntrinsicValue columns to the DataFrame,
    computed from fundamentals already present (EPS, BVPS, FCF). Does NOT
    require a Price column — intrinsic value is computable from SEC data
    alone.

    Vectorized for the Graham Number path (EPS and BVPS are column
    operations).  DCF still iterates because the two-stage model's
    year-by-year projection doesn't vectorize cleanly, but we use a fast
    list-comprehension path.
    """
    df = df.copy()

    # Prefer cover-page shares outstanding (dei:EntityCommonStockSharesOutstanding)
    # over the balance-sheet figure — the cover-page figure is typically
    # weeks-to-months more current.  Fall back to balance-sheet per-row when
    # the cover-page tag isn't present (older/smaller filers often don't tag it).
    cover_shares = df.get("CommonStockSharesOutstandingCoverPage")
    if cover_shares is not None:
        shares = cover_shares.fillna(df.get("CommonStockSharesOutstanding"))
    else:
        shares = df.get("CommonStockSharesOutstanding")
    shares = shares.astype(float)
    ni = df["NetIncomeLoss"].astype(float)
    equity = df["StockholdersEquity"].astype(float)
    fcf_col = df["FCF"].astype(float)

    # --- Graham Number (vectorized) ---
    eps = np.where(
        shares.notna() & (shares != 0) & ni.notna(),
        ni / shares,
        np.nan,
    )
    bvps = np.where(
        shares.notna() & (shares != 0) & equity.notna(),
        equity / shares,
        np.nan,
    )
    valid_gn = (eps > 0) & (bvps > 0) & ~np.isnan(eps) & ~np.isnan(bvps)
    gn = np.full(len(df), np.nan)
    gn[valid_gn] = np.sqrt(22.5 * eps[valid_gn] * bvps[valid_gn])
    df["GrahamNumber"] = gn

    # --- DCF Intrinsic Value (per-row loop — the projection math doesn't
    #      vectorize cleanly, but we avoid df.iterrows() overhead) ---
    shares_arr = shares.to_numpy()
    fcf_arr = fcf_col.to_numpy()
    dcf_values = np.full(len(df), np.nan)
    for i in range(len(df)):
        dcf_values[i] = compute_dcf_intrinsic_value_per_share(
            fcf=fcf_arr[i] if not np.isnan(fcf_arr[i]) else None,
            shares_outstanding=shares_arr[i] if not np.isnan(shares_arr[i]) else None,
            growth_rate=growth_rate,
            discount_rate=discount_rate,
            terminal_growth_rate=terminal_growth_rate,
            projection_years=projection_years,
        )
    df["DCFIntrinsicValue"] = dcf_values

    return df


# ---------------------------------------------------------------------------
# Margin of Safety — reads pre-computed columns (does NOT recompute)
# ---------------------------------------------------------------------------

def compute_margin_of_safety(
    df: pd.DataFrame,
    growth_rate: float = 0.05,
    discount_rate: float = 0.10,
    terminal_growth_rate: float = 0.025,
    projection_years: int = 5,
) -> pd.DataFrame:
    """Adds MarginOfSafetyGraham and MarginOfSafetyDCF columns.  Requires
    'Price' to be present (for the comparison denominator) and relies on
    'GrahamNumber' and 'DCFIntrinsicValue' columns being already computed
    by apply_valuation (called earlier in screen_data_from_cache).

    Margin of safety = (IntrinsicValue − Price) / Price.
    Positive = undervalued, negative = overvalued.

    This function no longer recomputes EPS/BVPS/DCF — it reads the columns
    apply_valuation wrote, so there is exactly one source of truth for
    intrinsic values and no risk of the MoS numerator diverging from the
    displayed Valuation figure.
    """
    if "Price" not in df.columns:
        return df

    df = df.copy()

    price = pd.to_numeric(df["Price"], errors="coerce")

    # Graham MoS
    if "GrahamNumber" in df.columns:
        gn_col = pd.to_numeric(df["GrahamNumber"], errors="coerce")
        valid = gn_col.notna() & price.notna() & (price != 0)
        df["MarginOfSafetyGraham"] = np.where(
            valid, (gn_col - price) / price, np.nan
        )
    else:
        df["MarginOfSafetyGraham"] = np.nan

    # DCF MoS
    if "DCFIntrinsicValue" in df.columns:
        dcf_col = pd.to_numeric(df["DCFIntrinsicValue"], errors="coerce")
        valid = dcf_col.notna() & price.notna() & (price != 0)
        df["MarginOfSafetyDCF"] = np.where(
            valid, (dcf_col - price) / price, np.nan
        )
    else:
        df["MarginOfSafetyDCF"] = np.nan

    return df
