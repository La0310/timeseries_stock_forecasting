"""
scenario_router.py
POST /api/scenario — What-If Price/Promotion Scenario Simulator.

Orchestrates the full pipeline for both a baseline and a modified-demand
scenario, using the same verified services as /api/forecast unchanged.

Spec: Addendum — What-If Price/Promotion Scenario Simulator (v1) §3–4.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from backend.services.demand_classifier import DemandClassifier
from backend.services.forecast_engine import ForecastEngine
from backend.services.inventory_recommender import InventoryRecommender
from backend.services.model_router import ModelRouter
from backend.services.purchase_order_service import PurchaseOrderService
from backend.services.scenario_engine import (
    ASSUMPTION_NOTE,
    apply_multiplier,
    build_forecast_summary,
    compute_demand_multiplier,
    get_elasticity,
)

router = APIRouter()

# ---------------------------------------------------------------------------
# Data loading — same pattern as forecast_router.py
# ---------------------------------------------------------------------------

_DATA_PATH = Path(__file__).parent.parent / "data" / "demo_inventory.csv"
_df: pd.DataFrame | None = None


def _load_df() -> pd.DataFrame:
    global _df
    if _df is None:
        _df = pd.read_csv(_DATA_PATH, parse_dates=["Date"])
    return _df


def _get_sku_meta(sku_id: str) -> dict:
    """Returns SKU-level metadata from the first matching row."""
    df = _load_df()
    rows = df[df["Product ID"] == sku_id]
    if rows.empty:
        raise HTTPException(status_code=404, detail=f"SKU '{sku_id}' not found")
    first = rows.iloc[0]
    return {
        "product_name": first["Product Name"],
        "category": first["Category"],
        "pack_size": int(first["Pack Size"]),
        "lead_time_days": int(first["Lead Time Days"]),
        "current_stock": int(first["Current Stock"]),
    }


def _get_sales_history(sku_id: str) -> pd.Series:
    """Returns the daily Units Sold series for this SKU, sorted by date."""
    df = _load_df()
    rows = df[df["Product ID"] == sku_id].sort_values("Date")
    if rows.empty:
        raise HTTPException(status_code=404, detail=f"SKU '{sku_id}' not found")
    return rows["Units Sold"].reset_index(drop=True).astype(float)


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------


class ScenarioRequest(BaseModel):
    sku_id: str
    price_change_pct: float          # signed %; +10 = 10 % price increase
    promotion: bool
    horizon_days: int = 28


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------


@router.post("/api/scenario")
def post_scenario(req: ScenarioRequest) -> dict:
    """
    Runs the full pipeline twice — once for baseline, once for the scenario —
    and returns both sets of forecast_summary and replenishment results.

    Pipeline (spec §3):
      1.  Load sales history + metadata.
      2.  DemandClassifier.analyze()                → demand_analysis / daily_sales_std
      3.  ForecastEngine.run_backtest() × 6         → wape_dict
      4.  ModelRouter.route()                       → selected_model
      5.  PurchaseOrderService.get_on_order_stock() → on_order_stock
      6.  ForecastEngine.generate_quantiles()       → baseline quantiles (P50)
      7.  scenario_engine: elasticity lookup + demand_multiplier
      8.  apply_multiplier(baseline_p50)            → scenario_p50
      9.  InventoryRecommender.calculate() × 2:
            a. baseline   — daily_forecast = baseline_p50
            b. scenario   — daily_forecast = scenario_p50
      10. Assemble and return response schema per spec §4.

    Existing pipeline services are called UNCHANGED; no logic is reimplemented.
    """
    sku_id        = req.sku_id
    horizon_days  = req.horizon_days
    price_chg_pct = req.price_change_pct
    promotion     = req.promotion

    # ── 1. Load data ────────────────────────────────────────────────────────
    history = _get_sales_history(sku_id)
    meta    = _get_sku_meta(sku_id)
    category = meta["category"]

    # ── 2. Demand classification (daily_sales_std computed here and only here)
    demand_analysis  = DemandClassifier.analyze(history)
    daily_sales_std  = demand_analysis["daily_sales_std"]

    # ── 3. Walk-forward backtest for all 6 candidate models ─────────────────
    wape_dict = {
        model: ForecastEngine.run_backtest(history, model)
        for model in ModelRouter.MODEL_REGISTRY
    }

    # ── 4. Model selection ──────────────────────────────────────────────────
    routing        = ModelRouter.route(demand_analysis, wape_dict)
    selected_model = routing["selected_model"]  # None if all scores are null

    # ── 5. On-order stock from the in-memory ledger ─────────────────────────
    on_order_stock = PurchaseOrderService.get_on_order_stock(sku_id)

    # ── 6. Quantile forecast → baseline P50 ─────────────────────────────────
    quantiles    = ForecastEngine.generate_quantiles(
        history, selected_model, horizon_days, daily_sales_std
    )
    baseline_p50 = quantiles["P50"]

    # ── 7. Elasticity + demand multiplier (spec §3 step 3) ──────────────────
    elasticity       = get_elasticity(category)
    demand_multiplier = compute_demand_multiplier(price_chg_pct, promotion, elasticity)

    # ── 8. Scenario P50 ─────────────────────────────────────────────────────
    scenario_p50 = apply_multiplier(baseline_p50, demand_multiplier)

    # ── 9a. Baseline replenishment ──────────────────────────────────────────
    baseline_rep_raw = InventoryRecommender.calculate(
        current_stock   = meta["current_stock"],
        on_order_stock  = on_order_stock,
        category        = category,
        lead_time_days  = meta["lead_time_days"],
        daily_sales_std = daily_sales_std,
        daily_forecast  = baseline_p50,
        pack_size       = meta["pack_size"],
    )
    # Strip internal-only field (matches forecast_router convention)
    baseline_rep = {k: v for k, v in baseline_rep_raw.items() if k != "unrounded_order"}

    # ── 9b. Scenario replenishment ──────────────────────────────────────────
    scenario_rep_raw = InventoryRecommender.calculate(
        current_stock   = meta["current_stock"],
        on_order_stock  = on_order_stock,
        category        = category,
        lead_time_days  = meta["lead_time_days"],
        daily_sales_std = daily_sales_std,
        daily_forecast  = scenario_p50,
        pack_size       = meta["pack_size"],
    )
    scenario_rep = {k: v for k, v in scenario_rep_raw.items() if k != "unrounded_order"}

    # ── 10. Assemble response (spec §4) ─────────────────────────────────────
    return {
        "sku_id":             sku_id,
        "price_change_pct":   price_chg_pct,
        "promotion_applied":  promotion,
        "elasticity_used":    elasticity,
        "demand_multiplier":  round(demand_multiplier, 4),
        "assumption_note":    ASSUMPTION_NOTE,
        "baseline": {
            "forecast_summary": build_forecast_summary(baseline_p50),
            "replenishment":    baseline_rep,
        },
        "scenario": {
            "forecast_summary": build_forecast_summary(scenario_p50),
            "replenishment":    scenario_rep,
        },
    }
