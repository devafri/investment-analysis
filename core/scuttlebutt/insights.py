"""Pattern recognition across scuttlebutt notes — cross-company insights,
sentiment trends, and emerging themes."""

from collections import Counter
from typing import Dict, List

from core.scuttlebutt.models import Conviction, ScuttlebuttNote
from core.scuttlebutt.tags import get_tag_label


def tag_sentiment_summary(notes: List[ScuttlebuttNote]) -> Dict[str, dict]:
    """For each tag, count bullish / bearish / neutral notes.

    Returns ``{tag: {"bullish": N, "bearish": N, "neutral": N, "label": str}}``.
    """
    summary: Dict[str, dict] = {}
    for note in notes:
        for tag in note.tags:
            if tag not in summary:
                summary[tag] = {
                    "bullish": 0, "bearish": 0, "neutral": 0,
                    "label": get_tag_label(tag),
                }
            if note.conviction == Conviction.BULLISH:
                summary[tag]["bullish"] += 1
            elif note.conviction == Conviction.BEARISH:
                summary[tag]["bearish"] += 1
            else:
                summary[tag]["neutral"] += 1
    return summary


def conviction_distribution(notes: List[ScuttlebuttNote]) -> Dict[str, int]:
    """Return counts by conviction level."""
    dist = {"bullish": 0, "bearish": 0, "neutral": 0}
    for note in notes:
        dist[note.conviction.value] += 1
    return dist


def recent_activity(notes: List[ScuttlebuttNote], top_n: int = 10) -> List[dict]:
    """Return the *top_n* most recently updated notes as lightweight dicts
    suitable for template rendering."""
    sorted_notes = sorted(
        notes, key=lambda n: n.updated_at or n.created_at, reverse=True,
    )
    return [
        {
            "title": n.title,
            "company_name": n.company_name or f"CIK {n.company_cik}",
            "company_cik": n.company_cik,
            "category": n.category.value,
            "conviction": n.conviction.value,
            "source_quality": n.source_quality,
            "tags": [get_tag_label(t) for t in n.tags],
        }
        for n in sorted_notes[:top_n]
    ]
