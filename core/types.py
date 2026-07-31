"""Shared TypedDicts and type aliases used across the app.

Kept in one place so route handlers, screening logic, and templates all
agree on what each data structure looks like — and so IDE autocompletion
works across module boundaries.
"""

from typing import Any, Dict, List, Optional, TypedDict


class IngestState(TypedDict, total=False):
    running: bool
    started_at: Optional[float]
    total_sources: int
    current_source: str
    completed_sources: int
    total_filings: int
    sources_done: List[Dict[str, Any]]
    error: Optional[str]
    complete: bool


class ThresholdConfig(TypedDict, total=False):
    min_roic: float
    min_operating_margin: float
    max_debt_to_equity: float
    min_interest_coverage: float
    min_cfo_to_ni: float
    min_revenue_growth: float
    min_fscore: int
    require_positive_ni: bool
    major_exchanges_only: bool
    exclude_financials: bool
    # DCF assumptions
    growth_rate: float
    discount_rate: float
    terminal_growth_rate: float
    projection_years: int


class RowContext(TypedDict, total=False):
    row: Dict[str, Any]
    display_ticker: str
    cik: str
    company_url: str
    magic_formula_rank: Optional[int]
    is_true_magic_formula: bool
    roic: Optional[float]
    operating_margin: Optional[float]
    debt_to_equity: Optional[float]
    interest_coverage: Optional[float]
    cfo_to_ni: Optional[float]
    # Price & intrinsic value
    price: Optional[float]
    graham_number: Optional[float]
    dcf_intrinsic_value: Optional[float]
    dcf_delta: Optional[float]
    graham_delta: Optional[float]
    margin_of_safety_graham: Optional[float]
    margin_of_safety_dcf: Optional[float]
    # Growth
    revenue_growth: Optional[float]
    revenue_growth_3yr: Optional[float]
    fcf_growth: Optional[float]
    # F-Score
    fscore: Optional[int]
    # Multiples
    pe_ratio: Optional[float]
    pb_ratio: Optional[float]
    ev_to_ebit: Optional[float]
    p_fcf: Optional[float]
    # Sector
    sector: Optional[str]
    roic_sector_pct: Optional[float]
    om_sector_pct: Optional[float]
    # Trend
    roic_trend: str
    margin_trend: str
    # Watchlist
    added_at: Optional[str]
    added_price: Optional[float]
    current_price: Optional[float]
    price_change: Optional[float]
    price_change_pct: Optional[float]


class IngestLogEntry(TypedDict):
    source_name: str
    filings_added: int
    ingested_at: str


class Pagination(TypedDict):
    page: int
    total_pages: int
    page_size: int
    total_rows: int


class ScreenInfo(TypedDict, total=False):
    thresholds: ThresholdConfig
    errors: List[str]
    diagnostics: Dict[str, int]
    summary: Optional[Dict[str, Any]]
