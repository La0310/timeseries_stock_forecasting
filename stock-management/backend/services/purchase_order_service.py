"""
purchase_order_service.py
In-memory on-order ledger for the demo.
Spec: 02_pipeline_workflow_v3.md §2.5 / 01_product_spec_v3.md §4 Guardrail 5

This is the ONLY module that manages on_order_stock. inventory_recommender.py
reads on-order quantities from here but does not write or mutate them.

A production build would back this with a real purchase-order table; the
interface is written so that swap is a storage-layer change only, not a
call-site change.
"""

from collections import defaultdict


class PurchaseOrderService:
    _on_order_by_sku: dict[str, int] = defaultdict(int)
    _po_sequence: int = 0

    @classmethod
    def record_purchase_order(cls, sku_id: str, quantity: int) -> dict:
        """
        Records a new purchase order for sku_id and returns the ledger entry.

        Returns dict with: po_id, sku_id, quantity_ordered, on_order_stock
        on_order_stock is the cumulative total for this SKU after recording.
        """
        cls._po_sequence += 1
        po_id = f"PO-2026-{cls._po_sequence:06d}"
        cls._on_order_by_sku[sku_id] += quantity
        return {
            "po_id": po_id,
            "sku_id": sku_id,
            "quantity_ordered": quantity,
            "on_order_stock": cls._on_order_by_sku[sku_id],
        }

    @classmethod
    def get_on_order_stock(cls, sku_id: str) -> int:
        """Returns the current cumulative on-order quantity for sku_id (0 if none)."""
        return cls._on_order_by_sku[sku_id]

    @classmethod
    def reset(cls) -> None:
        """Resets the ledger — for testing only. Not called in production paths."""
        cls._on_order_by_sku = defaultdict(int)
        cls._po_sequence = 0
