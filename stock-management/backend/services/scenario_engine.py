"""
scenario_engine.py
What-If Price/Promotion Scenario Simulator — elasticity assumptions and
scenario demand-multiplier math.

## Design note (WHY assumptions, not regression)

check_scenario_columns.py confirmed that Price, Discount, Holiday/Promotion,
Competitor Pricing, and Weather Condition are ALL constant per SKU across
every one of the 365 history rows in every archetype.  Zero historical
variance means no regression technique can extract an empirical
price-to-demand relationship from this dataset.

Decision: use literature-informed, per-category price elasticity values —
the same pattern already used for SERVICE_LEVEL_Z in inventory_recommender.py.
The assumption is surfaced explicitly in every response via assumption_note;
it is never presented as an empirically derived number.

Spec: Addendum — What-If Price/Promotion Scenario Simulator (v1) §2–3.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Elasticity assumptions — single source of truth for this feature.
# Values are literature-informed per-category defaults.
# ---------------------------------------------------------------------------

CATEGORY_ELASTICITY: dict[str, float] = {
    "Beverage": -0.6,         # inelastic daily consumable
    "Office": -0.7,           # moderately inelastic staple supply
    "Hardware": -1.2,         # elastic: substitutes available, buyer-driven
    "MRO / Parts": -0.4,      # very inelastic: often sole-source, ops-critical
    "Seasonal overlay": -1.8, # highly elastic: discretionary, promo-sensitive
}
DEFAULT_ELASTICITY: float = -1.0   # unit-elastic fallback for unlisted categories

# Flat promotion uplift — not category-differentiated per spec §2.
PROMOTION_UPLIFT: float = 1.20    # +20 % demand when a promotion is active

ASSUMPTION_NOTE: str = (
    "Elasticity is a literature-informed assumption per category. "
    "This demo dataset has no historical price variation to estimate it "
    "from empirically."
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_elasticity(category: str) -> float:
    """Return the price elasticity for *category*, or DEFAULT_ELASTICITY."""
    return CATEGORY_ELASTICITY.get(category, DEFAULT_ELASTICITY)


def compute_demand_multiplier(
    price_change_pct: float,
    promotion: bool,
    elasticity: float,
) -> float:
    """
    Compute the scenario demand multiplier per spec §3 step 3.

    multiplier = 1 + (elasticity * price_change_pct / 100)
    if promotion: multiplier *= 1.20
    multiplier = max(0.0, multiplier)   # demand can never go negative

    Parameters
    ----------
    price_change_pct : signed percentage change in price
                       (+10 means a 10 % price increase)
    promotion        : whether a promotion is simultaneously active
    elasticity       : signed price elasticity (negative for normal goods)

    Returns
    -------
    float >= 0.0
    """
    multiplier = 1.0 + (elasticity * price_change_pct / 100.0)
    if promotion:
        multiplier *= PROMOTION_UPLIFT
    return max(0.0, multiplier)


def apply_multiplier(baseline_p50: list[float], multiplier: float) -> list[float]:
    """
    Scale each day of *baseline_p50* by *multiplier*.

    Returns a new list; baseline is never mutated.
    """
    return [d * multiplier for d in baseline_p50]


def build_forecast_summary(p50: list[float]) -> dict[str, int]:
    """
    Integer sums of *p50* over 7 / 14 / 28-day windows — same logic as
    forecast_router.py step 8, kept here to avoid importing across layers.
    """
    return {
        "7_days":  int(round(sum(p50[:7]))),
        "14_days": int(round(sum(p50[:14]))),
        "28_days": int(round(sum(p50[:28]))),
    }
