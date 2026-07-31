"""Data models for qualitative scuttlebutt research notes.

Scuttlebutt is Phil Fisher's term for gathering qualitative intelligence
about a company — talking to customers, competitors, suppliers, ex-employees,
and observing products firsthand.  These models capture those observations
in a structured way that can be tagged, searched, and triangulated with
quantitative signals.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import List, Optional


class NoteCategory(str, Enum):
    PRODUCT = "product"            # Product / service quality, roadmap
    CUSTOMER = "customer"          # Customer feedback, satisfaction
    COMPETITOR = "competitor"      # Competitive positioning, threats
    MANAGEMENT = "management"      # Management quality, culture, ethics
    INDUSTRY = "industry"          # Industry trends, regulatory changes
    FINANCIAL = "financial"        # Anecdotal financial data points
    OTHER = "other"


class Conviction(str, Enum):
    BULLISH = "bullish"            # Positive signal for the investment thesis
    BEARISH = "bearish"            # Negative signal
    NEUTRAL = "neutral"            # Informational, no directional signal


@dataclass
class ScuttlebuttNote:
    """A single qualitative observation about a company or industry."""

    id: str                         # Unique ID (e.g. UUID)
    company_cik: Optional[str]      # CIK if tied to a specific company
    company_name: Optional[str]     # Human-readable company name
    category: NoteCategory
    conviction: Conviction
    title: str                      # Short summary line
    body: str                       # Full observation text
    source_type: str = ""           # e.g. "customer_interview", "store_visit",
                                    #       "ex_employee", "industry_report"
    source_quality: int = 3         # 1 (rumour) to 5 (first-hand verified)
    tags: List[str] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = ""

    def __post_init__(self):
        if not self.updated_at:
            self.updated_at = self.created_at


@dataclass
class CompanyAssessment:
    """Aggregated qualitative assessment for a company."""

    company_cik: str
    company_name: str
    bullish_notes: int = 0
    bearish_notes: int = 0
    neutral_notes: int = 0
    avg_source_quality: float = 0.0
    top_tags: List[str] = field(default_factory=list)
    last_updated: str = ""
    summary: str = ""               # Free-text analyst summary

    @property
    def net_sentiment(self) -> int:
        """Net bullish minus bearish count."""
        return self.bullish_notes - self.bearish_notes

    @property
    def conviction_ratio(self) -> float:
        """Bullish / (bullish + bearish), or 0.5 if no directional notes."""
        total = self.bullish_notes + self.bearish_notes
        if total == 0:
            return 0.5
        return self.bullish_notes / total
