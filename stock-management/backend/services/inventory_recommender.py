"""
inventory_recommender.py
Safety stock, ROP, Inventory Position, IP-gated order quantity, and status.
Spec: 02_pipeline_workflow_v3.md §2.4 / 03_addendum_spec_v3.md §3.2, §3.3

CRITICAL CONSTRAINTS:
  - daily_sales_std is PASSED IN from demand_classifier; never recomputed here.
  - Status is computed exactly once (classify_status) from IP and ROP.
    IP = OnHand + OnOrder. Raw on-hand stock is never used alone for status.
  - SERVICE_LEVEL_Z lives here and only here (03 §4 item 8).
"""

import numpy as np

# Category service levels. Single source of truth per spec §2.4 / 03 §3.2.
SERVICE_LEVEL_Z: dict[str, float] = {
    "Beverage": 1.65,          # 95% — fast-turning, short lead time, low stockout cost
    "Office": 1.65,            # 95% — fast-turning consumables, same reasoning
    "Hardware": 1.96,          # 97.5% — higher unit cost, promo-driven demand spikes
    "MRO / Parts": 2.33,       # 99% — long lead time; stockout can stall buyer's ops
    "Seasonal overlay": 1.28,  # 90% — overstock/markdown risk exceeds stockout risk
}
DEFAULT_Z = 1.65

LOW_THRESHOLD_BY_CATEGORY: dict[str, float] = {
    "Beverage":         1.2,   # calibrated from M5: high-frequency, low stockout risk
    "Hardware":         1.2,
    "MRO / Parts":      1.2,
    "Office":           1.2,
    "Seasonal overlay": 1.2,
}
DEFAULT_LOW_THRESHOLD = 1.5   # fallback for categories not in the map


class InventoryRecommender:
    @staticmethod
    def service_level_z(category: str) -> float:
        """Looks up Z-score for the given category; falls back to DEFAULT_Z."""
        return SERVICE_LEVEL_Z.get(category, DEFAULT_Z)

    @staticmethod
    def classify_status(
        inventory_position: float,
        reorder_point: float,
        category: str = "",
    ) -> str:
        """
        Single-basis status rule (spec §2.4 / 01 §3 Status badge):
          Critical : IP <= ROP
          Low      : ROP < IP <= low_thresh * ROP
          Healthy  : IP > low_thresh * ROP

        `low_thresh` is looked up from LOW_THRESHOLD_BY_CATEGORY (calibrated
        from M5 catalog data). Falls back to DEFAULT_LOW_THRESHOLD (1.5) for
        unknown categories.

        This function is the sole source of 'status' and 'new_status' across
        Screens 1, 2, and 3. Frontend never recomputes or infers status.
        """
        low_thresh = LOW_THRESHOLD_BY_CATEGORY.get(category, DEFAULT_LOW_THRESHOLD)
        if inventory_position <= reorder_point:
            return "Critical"
        if inventory_position <= low_thresh * reorder_point:
            return "Low"
        return "Healthy"

    @staticmethod
    def calculate(
        current_stock: int,
        on_order_stock: int,
        category: str,
        lead_time_days: int,
        daily_sales_std: float,
        daily_forecast: list[float],
        pack_size: int = 1,
    ) -> dict:
        """
        Computes full replenishment recommendation for one SKU.

        Parameters
        ----------
        current_stock   : raw on-hand units
        on_order_stock  : open PO units, sourced from PurchaseOrderService
        category        : SKU category string, used for Z lookup
        lead_time_days  : supplier turnaround in days
        daily_sales_std : demand_analysis.daily_sales_std — passed through,
                          NEVER recomputed here (spec §2.1 Single-Source rule)
        daily_forecast  : P50 daily point forecasts from ForecastEngine
        pack_size       : minimum order multiple (case pack)

        Returns a dict whose keys match the spec §3.1 'replenishment' schema.
        """
        z_score = InventoryRecommender.service_level_z(category)
        inventory_position = current_stock + on_order_stock  # IP = OnHand + OnOrder

        lead_demand = float(np.sum(daily_forecast[:lead_time_days]))
        total_horizon_demand = float(np.sum(daily_forecast))

        # SS = ceil(Z × σ_daily × √L) — uses passed-in daily_sales_std
        safety_stock = int(np.ceil(z_score * daily_sales_std * np.sqrt(lead_time_days)))
        reorder_point = int(np.ceil(lead_demand + safety_stock))

        # IP-gated order quantity (spec §2.4 / 03 §3.3)
        if inventory_position > reorder_point:
            unrounded_order = 0  # IP-gate: lead-time buffer is already covered
        else:
            unrounded_order = max(
                0,
                int(np.ceil((total_horizon_demand + safety_stock) - inventory_position)),
            )

        # Pack-size rounding (spec §3.3)
        if pack_size > 1 and unrounded_order > 0:
            cases_needed = int(np.ceil(unrounded_order / pack_size))
            recommended_order = cases_needed * pack_size
        else:
            cases_needed = unrounded_order
            recommended_order = unrounded_order

        return {
            "current_stock": current_stock,
            "on_order_stock": on_order_stock,
            "inventory_position": inventory_position,
            "lead_time_days": lead_time_days,
            "lead_time_demand": round(lead_demand, 1),
            "service_level_z": z_score,
            "safety_stock": safety_stock,
            "reorder_point": reorder_point,
            "unrounded_order": unrounded_order,
            "pack_size": pack_size,
            "recommended_cases": cases_needed,
            "recommended_order": recommended_order,
            "status": InventoryRecommender.classify_status(
                inventory_position, reorder_point, category
            ),
        }
