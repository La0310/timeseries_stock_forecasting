"""
forecast_router.py
POST /api/forecast — orchestrates the demand analysis → model routing → forecast
→ replenishment pipeline and returns the spec §3.1 JSON schema.

Spec: 02_pipeline_workflow_v3.md §3.1 / 04_patch_notes_v3.1 Patch 2
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

router = APIRouter()

# Path to demo data — resolved relative to this file's package root
_DATA_PATH = Path(__file__).parent.parent / "data" / "demo_inventory.csv"

# Cache the CSV once at import time (tiny dataset, safe to hold in memory)
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


class ForecastRequest(BaseModel):
    sku_id: str
    horizon_days: int = 28


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------


@router.post("/api/forecast")
def post_forecast(req: ForecastRequest) -> dict:
    """
    Runs the full demand → routing → forecast → replenishment pipeline.

    Pipeline (spec §3.1):
      1. Load sales history
      2. DemandClassifier.analyze()  → demand_analysis (incl. daily_sales_std)
      3. ForecastEngine.run_backtest() for all 6 models → WAPE dict
      4. ModelRouter.route()          → routing (selected_model, scores, justification)
      5. PurchaseOrderService.get_on_order_stock() → on_order_stock
      6. ForecastEngine.generate_quantiles() using daily_sales_std from step 2
         and selected_model (or None→MA fallback) — daily_forecast = P50
      7. InventoryRecommender.calculate() — daily_forecast = P50 from step 6
      8. Assemble response with top-level limited_history flag (Patch v3.1)

    Field-name contract: no key is renamed between service return values
    and the JSON response (spec §3 / 03 §4 item 6).
    """
    sku_id = req.sku_id
    horizon_days = req.horizon_days

    # 1. Load history and metadata
    history = _get_sales_history(sku_id)
    meta = _get_sku_meta(sku_id)

    # 2. Demand classification (daily_sales_std computed here and only here)
    demand_analysis = DemandClassifier.analyze(history)
    daily_sales_std = demand_analysis["daily_sales_std"]

    # 3. Walk-forward backtest for all 6 models
    wape_dict = {
        model: ForecastEngine.run_backtest(history, model)
        for model in ModelRouter.MODEL_REGISTRY
    }

    # 4. Model selection (score-driven, no hardcoded mapping)
    routing = ModelRouter.route(demand_analysis, wape_dict)
    selected_model = routing["selected_model"]  # None if all scores are null

    # limited_history: fewer than 14 records OR every model scored null
    # (spec §3.1 Patch v3.1 / 03 §4 item 4)
    all_null = all(w is None for w in wape_dict.values())
    limited_history: bool = len(history) < 14 or all_null

    # 5. On-order stock from the in-memory ledger
    on_order_stock = PurchaseOrderService.get_on_order_stock(sku_id)

    # 6. Quantile forecast
    #    If selected_model is None (all-null scores), _model_point_forecast
    #    falls through to the moving-average baseline (model=None path).
    quantiles = ForecastEngine.generate_quantiles(
        history, selected_model, horizon_days, daily_sales_std
    )
    p50 = quantiles["P50"]  # daily_forecast passed to InventoryRecommender

    # 7. Replenishment calculation
    #    daily_forecast = P50 (patch note 04 §item 3: "daily_forecast is P50")
    replenishment_raw = InventoryRecommender.calculate(
        current_stock=meta["current_stock"],
        on_order_stock=on_order_stock,
        category=meta["category"],
        lead_time_days=meta["lead_time_days"],
        daily_sales_std=daily_sales_std,   # from step 2, not recomputed
        daily_forecast=p50,
        pack_size=meta["pack_size"],
    )
    replenishment = {k: v for k, v in replenishment_raw.items() if k != "unrounded_order"}
    
    # 8. Forecast summary (integer sums of P50 over 7/14/28 day windows)
    forecast_summary = {
        "7_days": int(round(sum(p50[:7]))),
        "14_days": int(round(sum(p50[:14]))),
        "28_days": int(round(sum(p50[:28]))),
    }

    # Assemble exact response schema per spec §3.1
    return {
        "sku_id": sku_id,
        "limited_history": limited_history,          # Patch v3.1 §3.1
        "product_name": meta["product_name"],
        "category": meta["category"],
        "demand_analysis": demand_analysis,
        "routing": routing,
        "forecast": {
            "summary": forecast_summary,
            "quantiles": quantiles,
        },
        "replenishment": replenishment,
    }
