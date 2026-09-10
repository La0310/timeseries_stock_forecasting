"""
purchase_order_router.py
POST /api/purchase-order — records a PO, recomputes post-PO Inventory Position
and new_status, returns spec §3.2 JSON schema.

Spec: 02_pipeline_workflow_v3.md §3.2 / 01_product_spec_v3.md §3 Screen 3
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from backend.services.forecast_engine import ForecastEngine
from backend.services.inventory_recommender import InventoryRecommender
from backend.services.model_router import ModelRouter
from backend.services.demand_classifier import DemandClassifier
from backend.services.purchase_order_service import PurchaseOrderService

router = APIRouter()

_DATA_PATH = Path(__file__).parent.parent / "data" / "demo_inventory.csv"
_df: pd.DataFrame | None = None


def _load_df() -> pd.DataFrame:
    global _df
    if _df is None:
        _df = pd.read_csv(_DATA_PATH, parse_dates=["Date"])
    return _df


def _get_sku_meta(sku_id: str) -> dict:
    """Returns SKU-level metadata from the most recent matching row.
    Current Stock is time-varying day-to-day, so this must read the
    latest date, never an arbitrary/unsorted 'first' row."""
    df = _load_df()
    rows = df[df["Product ID"] == sku_id].sort_values("Date")
    if rows.empty:
        raise HTTPException(status_code=404, detail=f"SKU '{sku_id}' not found")
    latest = rows.iloc[-1]
    return {
        "product_name": latest["Product Name"],
        "category": latest["Category"],
        "pack_size": int(latest["Pack Size"]),
        "lead_time_days": int(latest["Lead Time Days"]),
        "current_stock": int(latest["Current Stock"]),
    }


def _get_sales_history(sku_id: str) -> pd.Series:
    df = _load_df()
    rows = df[df["Product ID"] == sku_id].sort_values("Date")
    if rows.empty:
        raise HTTPException(status_code=404, detail=f"SKU '{sku_id}' not found")
    return rows["Units Sold"].reset_index(drop=True).astype(float)


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------


class PORequest(BaseModel):
    sku_id: str
    quantity: int


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------


@router.post("/api/purchase-order")
def post_purchase_order(req: PORequest) -> dict:
    """
    Records a purchase order and returns the post-PO inventory state.

    Pipeline (spec §3.2):
      1. PurchaseOrderService.record_purchase_order() → po_id, new on_order_stock
      2. Re-run demand analysis + model routing to get current ROP
         (payload only carries sku_id + quantity; ROP must be re-derived)
      3. Compute post-PO inventory_position = current_stock + new on_order_stock
      4. InventoryRecommender.classify_status(ip, rop) → new_status
      5. Return spec §3.2 schema

    new_status is computed by the same classify_status() used everywhere else.
    The frontend renders it verbatim — it never assumes or hardcodes the
    before/after transition (Guardrail 6, 01 §4).
    """
    sku_id = req.sku_id
    quantity = req.quantity

    # 1. Record the PO and get the updated on-order stock
    po_record = PurchaseOrderService.record_purchase_order(sku_id, quantity)
    new_on_order = po_record["on_order_stock"]

    # 2. Re-derive ROP (need demand analysis + forecast for this SKU)
    meta = _get_sku_meta(sku_id)
    history = _get_sales_history(sku_id)

    demand_analysis = DemandClassifier.analyze(history)
    daily_sales_std = demand_analysis["daily_sales_std"]

    wape_dict = {
        model: ForecastEngine.run_backtest(history, model)
        for model in ModelRouter.MODEL_REGISTRY
    }
    routing = ModelRouter.route(demand_analysis, wape_dict)
    selected_model = routing["selected_model"]

    quantiles = ForecastEngine.generate_quantiles(
        history, selected_model, 28, daily_sales_std
    )
    p50 = quantiles["P50"]

    # Compute replenishment using the updated on_order_stock so ROP is current
    replenishment = InventoryRecommender.calculate(
        current_stock=meta["current_stock"],
        on_order_stock=new_on_order,
        category=meta["category"],
        lead_time_days=meta["lead_time_days"],
        daily_sales_std=daily_sales_std,
        daily_forecast=p50,
        pack_size=meta["pack_size"],
    )
    reorder_point = replenishment["reorder_point"]

    # 3-4. Post-PO Inventory Position and status
    current_stock = meta["current_stock"]
    inventory_position = current_stock + new_on_order
    new_status = InventoryRecommender.classify_status(inventory_position, reorder_point)

    # 5. Return exact spec §3.2 schema — no field renaming
    return {
        "po_id": po_record["po_id"],
        "sku_id": sku_id,
        "quantity_ordered": quantity,
        "current_stock": current_stock,
        "on_order_stock": new_on_order,
        "inventory_position": inventory_position,
        "reorder_point": reorder_point,
        "new_status": new_status,
    }
