"""Tag taxonomy for scuttlebutt research notes.

Tags provide a lightweight, user-extensible organising system.  Notes can
carry multiple tags, and the same tag can span companies — enabling
cross-company pattern recognition (e.g. "all companies tagged
'supply_chain_risk' with bearish notes last quarter").
"""

# Suggested starter tags — users can add their own via the UI.
PRESET_TAGS: dict[str, str] = {
    # Product & operations
    "product_quality":       "Product quality / reliability",
    "product_innovation":    "Innovation / R&D pipeline",
    "pricing_power":         "Pricing power / price increases",
    "supply_chain":          "Supply chain / logistics",
    # Competitive
    "moat_widening":         "Competitive moat widening",
    "moat_narrowing":        "Competitive moat narrowing",
    "market_share_gain":     "Market share gaining",
    "market_share_loss":     "Market share losing",
    "disruption_risk":       "Disruption / obsolescence risk",
    # Management & culture
    "strong_management":     "Strong management / execution",
    "weak_management":       "Weak management / execution",
    "talent_retention":      "Talent retention / hiring",
    "capital_allocation":    "Capital allocation quality",
    "insider_alignment":     "Insider alignment w/ shareholders",
    # Financial & risk
    "revenue_growth":        "Revenue growth observations",
    "margin_pressure":       "Margin pressure / cost inflation",
    "balance_sheet_risk":    "Balance sheet / leverage concerns",
    "regulatory_risk":       "Regulatory / political risk",
    "customer_concentration": "Customer concentration risk",
    # Industry
    "industry_tailwind":     "Industry tailwind",
    "industry_headwind":     "Industry headwind",
    "secular_trend":         "Secular trend (multi-year)",
}


def get_tag_label(tag: str) -> str:
    """Return a human-readable label for a tag, or the tag itself."""
    return PRESET_TAGS.get(tag, tag.replace("_", " ").title())
