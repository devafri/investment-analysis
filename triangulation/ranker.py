"""Multi-signal ranking — combine Value, Insider, and Scuttlebutt signals
into a unified investment idea score.

Each strategy contributes a 0–1 score.  The combined score is a weighted
average.  Ideas are ranked from strongest to weakest.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class SignalScore:
    """Score from one strategy for one company."""

    company_cik: str
    company_name: str

    # Individual strategy scores (0.0–1.0, higher = stronger investment case)
    value_score: float = 0.0        # Fundamentals: Magic Formula, F-Score, etc.
    insider_score: float = 0.0      # Insider: opportunistic buy intensity
    scuttlebutt_score: float = 0.0  # Qualitative: conviction ratio

    # Metadata for traceability
    value_detail: str = ""          # e.g. "ROIC 23%, F-Score 7, MF Rank 12"
    insider_detail: str = ""        # e.g. "3 opp buys, 0 opp sells this quarter"
    scuttlebutt_detail: str = ""    # e.g. "5 bullish, 1 bearish notes"

    @property
    def combined_score(self, weights: Optional[Dict[str, float]] = None) -> float:
        """Weighted average of available signals.

        Default weights: Value 0.40, Insider 0.35, Scuttlebutt 0.25.
        Signals with a score of 0 (no data) are excluded from the average;
        remaining weights are re-normalised.
        """
        w = weights or {"value": 0.40, "insider": 0.35, "scuttlebutt": 0.25}
        total = 0.0
        weight_sum = 0.0
        for key, score in [
            ("value", self.value_score),
            ("insider", self.insider_score),
            ("scuttlebutt", self.scuttlebutt_score),
        ]:
            if score > 0:
                total += score * w[key]
                weight_sum += w[key]
        return total / weight_sum if weight_sum > 0 else 0.0


def value_score_from_summary(summary: dict) -> float:
    """Convert a fundamentals summary dict into a 0–1 value score.

    Considers ROIC rank and F-Score.  Returns 0 if no data."""
    if not summary or summary.get("total_passed", 0) == 0:
        return 0.0
    score = 0.0
    # ROIC component: median ROIC across the universe
    median_roic = summary.get("median_roic") or 0
    score += min(median_roic / 0.20, 1.0) * 0.6  # 20% ROIC = full marks
    # F-Score component: not readily available in summary, use pass rate
    score += 0.4  # placeholder — detailed scores need per-company data
    return min(score, 1.0)


def insider_score_from_summary(summary: dict) -> float:
    """Convert an insider trading summary into a 0–1 insider score.

    Favours companies with net opportunistic buying."""
    if not summary or summary.get("total_trades", 0) == 0:
        return 0.0
    opp_buys = summary.get("opp_buys", 0)
    opp_sells = summary.get("opp_sells", 0)
    net = opp_buys - opp_sells
    total = opp_buys + opp_sells
    if total == 0:
        return 0.0 if opp_buys == 0 else 0.5
    # Net buy ratio normalised to 0–1
    net_ratio = (net / total + 1) / 2  # maps -1..+1 to 0..1
    return round(max(0.0, min(1.0, net_ratio)), 3)


@dataclass
class RankedIdea:
    """A ranked investment idea combining all available signals."""

    company_cik: str
    company_name: str
    ticker: str = ""
    signals: SignalScore = field(default_factory=lambda: SignalScore("", ""))
    combined_score: float = 0.0


def rank_ideas(
    value_data: Dict[str, dict],       # CIK → value stats
    insider_data: Dict[str, dict],     # CIK → insider stats
    scuttlebutt_data: Dict[str, dict], # CIK → scuttlebutt stats
    top_n: int = 50,
) -> List[RankedIdea]:
    """Build and rank a combined list of investment ideas.

    Companies appear only if they have at least one non-zero signal.
    """
    all_ciks = set(value_data.keys()) | set(insider_data.keys()) | set(
        scuttlebutt_data.keys()
    )
    ideas = []
    for cik in all_ciks:
        vs = value_score_from_summary(value_data.get(cik, {}))
        ins = insider_score_from_summary(insider_data.get(cik, {}))
        scut = scuttlebutt_data.get(cik, {})
        scut_score = (
            scut.get("conviction_ratio", 0.5) if scut else 0.0
        )

        name = (
            value_data.get(cik, {}).get("name", "")
            or insider_data.get(cik, {}).get("company_name", "")
            or scuttlebutt_data.get(cik, {}).get("company_name", "")
            or f"CIK {cik}"
        )

        signals = SignalScore(
            company_cik=cik,
            company_name=name,
            value_score=round(vs, 3),
            insider_score=round(ins, 3),
            scuttlebutt_score=round(scut_score, 3),
        )
        combined = signals.combined_score
        if combined > 0:
            ideas.append(RankedIdea(
                company_cik=cik,
                company_name=name,
                signals=signals,
                combined_score=round(combined, 3),
            ))

    ideas.sort(key=lambda i: i.combined_score, reverse=True)
    return ideas[:top_n]
