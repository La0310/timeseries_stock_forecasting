"""
purchase_order_router.py
Full PO lifecycle API — Create / Receive / Cancel / Get / List.

Endpoints
---------
POST   /api/purchase-order                    Create a new PO
GET    /api/purchase-orders                   List all POs for a SKU (?sku_id=)
GET    /api/purchase-order/{po_id}            Get a single PO by ID
POST   /api/purchase-order/{po_id}/receive    Receive full or partial stock
POST   /api/purchase-order/{po_id}/cancel     Cancel an open PO

Exception translation
---------------------
The service layer raises domain exceptions (backend.exceptions).
This router is the ONLY place that translates them to HTTP status codes:
  PONotFound          → 404
  POInvalidTransition → 409
  POInvalidQuantity   → 422

Spec: 02_pipeline_workflow_v3.md §3.2 / 01_product_spec_v3.md §3 Screen 3
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from backend.db.database import get_db
from backend.exceptions import POInvalidQuantity, POInvalidTransition, PONotFound
from backend.services.demand_classifier import DemandClassifier
from backend.services.forecast_engine import ForecastEngine
from backend.services.inventory_recommender import InventoryRecommender
from backend.services.model_router import ModelRouter
from backend.services.purchase_order_service import (
    po_cancel,
    po_create,
    po_get,
    po_list_by_sku,
    po_on_order_stock,
    po_receive,
)

router = APIRouter()

_DATA_PATH = Path(__file__).parent.parent / "data" / "demo_inventory.csv"
_df: pd.DataFrame | None = None


def _load_df() -> pd.DataFrame:
    global _df
    if _df is None:
        _df = pd.read_csv(_DATA_PATH, parse_dates=["Date"])
    return _df


def _get_sku_meta(sku_id: str) -> dict:
    """Returns SKU-level metadata from the most recent matching row."""
    df = _load_df()
    rows = df[df["Product ID"] == sku_id].sort_values("Date")
    if rows.empty:
        raise HTTPException(status_code=404, detail=f"SKU '{sku_id}' not found")
    latest = rows.iloc[-1]
    return {
        "product_name":   latest["Product Name"],
        "category":       latest["Category"],
        "pack_size":      int(latest["Pack Size"]),
        "lead_time_days": int(latest["Lead Time Days"]),
        "current_stock":  int(latest["Current Stock"]),
    }


def _get_sales_history(sku_id: str) -> pd.Series:
    df = _load_df()
    rows = df[df["Product ID"] == sku_id].sort_values("Date")
    if rows.empty:
        raise HTTPException(status_code=404, detail=f"SKU '{sku_id}' not found")
    return rows["Units Sold"].reset_index(drop=True).astype(float)


def _derive_replenishment(sku_id: str, on_order_stock: int) -> tuple[dict, dict]:
    """
    Re-derive demand analysis + model routing + replenishment for a SKU.
    Used by Create / Receive / Cancel to compute post-action inventory state.
    """
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

    quantiles = ForecastEngine.generate_quantiles(history, selected_model, 28, daily_sales_std)
    p50 = quantiles["P50"]

    replenishment = InventoryRecommender.calculate(
        current_stock=meta["current_stock"],
        on_order_stock=on_order_stock,
        category=meta["category"],
        lead_time_days=meta["lead_time_days"],
        daily_sales_std=daily_sales_std,
        daily_forecast=p50,
        pack_size=meta["pack_size"],
    )
    return replenishment, meta


# ---------------------------------------------------------------------------
# Exception → HTTP translation (only layer allowed to use HTTPException)
# ---------------------------------------------------------------------------

def _translate_po_exceptions(exc: Exception) -> HTTPException:
    """Convert a domain exception into the appropriate HTTPException."""
    if isinstance(exc, PONotFound):
        return HTTPException(status_code=404, detail=str(exc))
    if isinstance(exc, POInvalidTransition):
        return HTTPException(status_code=409, detail=str(exc))
    if isinstance(exc, POInvalidQuantity):
        return HTTPException(status_code=422, detail=str(exc))
    raise exc  # unknown exceptions propagate as 500


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class POCreateRequest(BaseModel):
    sku_id: str
    quantity: int = Field(..., gt=0, description="Units to order (must be > 0)")
    notes: Optional[str] = Field(None, description="Optional free-text notes")


class POReceiveRequest(BaseModel):
    quantity_received: int = Field(
        ..., gt=0, description="Units received in this delivery (must be > 0)"
    )


# ---------------------------------------------------------------------------
# POST /api/purchase-order  — Create
# ---------------------------------------------------------------------------

@router.post("/api/purchase-order")
def post_purchase_order(req: POCreateRequest, db: Session = Depends(get_db)) -> dict:
    """
    Records a purchase order and returns the post-PO inventory state.

    Pipeline (spec §3.2):
      1. po_create() → po_id, new on_order_stock
      2. Re-run demand analysis + model routing to get current ROP
      3. Compute post-PO inventory_position = current_stock + new on_order_stock
      4. InventoryRecommender.classify_status(ip, rop, category) → new_status
      5. Return spec §3.2 schema
    """
    try:
        po = po_create(db, req.sku_id, req.quantity, req.notes)
    except (PONotFound, POInvalidTransition, POInvalidQuantity) as exc:
        raise _translate_po_exceptions(exc)

    new_on_order = po["on_order_stock"]
    replenishment, meta = _derive_replenishment(req.sku_id, new_on_order)
    reorder_point = replenishment["reorder_point"]
    inventory_position = meta["current_stock"] + new_on_order
    new_status = InventoryRecommender.classify_status(
        inventory_position, reorder_point, meta["category"]
    )

    return {
        "po_id":             po["po_id"],
        "sku_id":            req.sku_id,
        "quantity_ordered":  req.quantity,
        "current_stock":     meta["current_stock"],
        "on_order_stock":    new_on_order,
        "inventory_position": inventory_position,
        "reorder_point":     reorder_point,
        "new_status":        new_status,
    }


# ---------------------------------------------------------------------------
# GET /api/purchase-orders?sku_id=  — List by SKU
# ---------------------------------------------------------------------------

@router.get("/api/purchase-orders")
def list_purchase_orders(sku_id: str, db: Session = Depends(get_db)) -> dict:
    """
    Returns all POs for a given SKU, newest first.
    on_order_stock is a SKU-level aggregate returned once at the top level,
    not duplicated on each individual PO row.
    """
    pos = po_list_by_sku(db, sku_id)
    on_order = po_on_order_stock(db, sku_id)
    return {
        "sku_id":           sku_id,
        "on_order_stock":   on_order,
        "purchase_orders":  pos,
        "total":            len(pos),
    }


# ---------------------------------------------------------------------------
# GET /api/purchase-order/{po_id}  — Get single PO
# ---------------------------------------------------------------------------

@router.get("/api/purchase-order/{po_id}")
def get_purchase_order(po_id: str, db: Session = Depends(get_db)) -> dict:
    """Returns a single PO by its po_id. 404 if not found."""
    try:
        return po_get(db, po_id)
    except PONotFound as exc:
        raise _translate_po_exceptions(exc)


# ---------------------------------------------------------------------------
# POST /api/purchase-order/{po_id}/receive  — Receive (full or partial)
# ---------------------------------------------------------------------------

@router.post("/api/purchase-order/{po_id}/receive")
def receive_purchase_order(
    po_id: str, req: POReceiveRequest, db: Session = Depends(get_db)
) -> dict:
    """
    Records delivery of `quantity_received` units against an open PO.

    - Partial receive: qty_received < remaining → status = PARTIALLY_RECEIVED
    - Full receive:    qty_received == remaining → status = RECEIVED
    - on_order_stock decreases by qty_received
    - Returns updated PO detail + post-receive inventory state
    """
    try:
        po = po_receive(db, po_id, req.quantity_received)
    except (PONotFound, POInvalidTransition, POInvalidQuantity) as exc:
        raise _translate_po_exceptions(exc)

    replenishment, meta = _derive_replenishment(po["sku_id"], po["on_order_stock"])
    reorder_point = replenishment["reorder_point"]
    inventory_position = meta["current_stock"] + po["on_order_stock"]
    new_status = InventoryRecommender.classify_status(
        inventory_position, reorder_point, meta["category"]
    )

    return {
        **po,
        "current_stock":      meta["current_stock"],
        "inventory_position": inventory_position,
        "reorder_point":      reorder_point,
        "new_status":         new_status,
    }


# ---------------------------------------------------------------------------
# POST /api/purchase-order/{po_id}/cancel  — Cancel
# ---------------------------------------------------------------------------

@router.post("/api/purchase-order/{po_id}/cancel")
def cancel_purchase_order(po_id: str, db: Session = Depends(get_db)) -> dict:
    """
    Cancels an open PO (CONFIRMED or PARTIALLY_RECEIVED).

    - on_order_stock decreases by the remaining (un-received) quantity.
    - Returns updated PO detail + post-cancel inventory state.
    - HTTP 409 if PO is already RECEIVED or CANCELLED.
    """
    try:
        po = po_cancel(db, po_id)
    except (PONotFound, POInvalidTransition, POInvalidQuantity) as exc:
        raise _translate_po_exceptions(exc)

    replenishment, meta = _derive_replenishment(po["sku_id"], po["on_order_stock"])
    reorder_point = replenishment["reorder_point"]
    inventory_position = meta["current_stock"] + po["on_order_stock"]
    new_status = InventoryRecommender.classify_status(
        inventory_position, reorder_point, meta["category"]
    )

    return {
        **po,
        "current_stock":      meta["current_stock"],
        "inventory_position": inventory_position,
        "reorder_point":      reorder_point,
        "new_status":         new_status,
    }
