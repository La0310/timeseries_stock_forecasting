"""
purchase_order_service.py
SQLite-backed PO lifecycle service: Create / Receive / Partial Receive / Cancel.

Public API (module-level functions, not a class)
-------------------------------------------------
po_create(db, sku_id, quantity, notes)    -> dict
po_receive(db, po_id, quantity_received)  -> dict
po_cancel(db, po_id)                      -> dict
po_get(db, po_id)                         -> dict
po_list_by_sku(db, sku_id)               -> list[dict]
po_on_order_stock(db, sku_id)            -> int

Why module functions, not a class?
  Every method was @staticmethod with no shared state — the class was just a
  namespace. Plain module functions are simpler, easier to import selectively,
  and don't require a class name prefix at every call site.

Design invariants
-----------------
- `on_order_stock` for a SKU is always a live SQL aggregate:
      SELECT SUM(qty_ordered - qty_received)
      WHERE sku_id = ? AND status IN ('CONFIRMED', 'PARTIALLY_RECEIVED')
  It is NEVER stored as a cached column or computed in Python — one SQL
  expression keeps it consistent and scales to any number of open POs.

- Status state machine:
      CONFIRMED ──► PARTIALLY_RECEIVED ──► RECEIVED
               └──► CANCELLED
  Cancellation is allowed from CONFIRMED or PARTIALLY_RECEIVED.
  A fully RECEIVED or already CANCELLED PO cannot be cancelled again.

- po_id format: PO-{YYYY}-{6-digit-autoincrement-id}, e.g. PO-2026-000001
  The sequence number comes from the DB-assigned `id` column (AUTOINCREMENT),
  guaranteeing uniqueness even under concurrent requests.

- Exceptions: this module raises only domain exceptions from backend.exceptions.
  It has NO dependency on FastAPI or HTTP. The router translates exceptions
  to HTTP status codes.
"""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import func
from sqlalchemy.orm import Session

from backend.db.models import POStatus, PurchaseOrder
from backend.exceptions import POInvalidQuantity, POInvalidTransition, PONotFound


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _now() -> datetime:
    return datetime.now(timezone.utc)


def _commit_and_refresh(db: Session, po: PurchaseOrder) -> None:
    """Commit the current transaction and reload `po` from the DB."""
    db.commit()
    db.refresh(po)


def _get_or_raise(db: Session, po_id: str) -> PurchaseOrder:
    """Return the PO row or raise PONotFound."""
    po = db.query(PurchaseOrder).filter(PurchaseOrder.po_id == po_id).first()
    if po is None:
        raise PONotFound(po_id)
    return po


# ---------------------------------------------------------------------------
# Create
# ---------------------------------------------------------------------------

def po_create(
    db: Session,
    sku_id: str,
    quantity: int,
    notes: str | None = None,
) -> dict:
    """
    Insert a new CONFIRMED purchase order.

    Uses the DB-assigned `id` (AUTOINCREMENT) to build `po_id` after insert,
    so the sequence is race-free even under concurrent requests.

    Returns the serialised PO dict plus the updated on_order_stock for the SKU.
    """
    now = _now()
    # Use a temporary placeholder so the NOT NULL constraint is satisfied
    # at flush time. We overwrite it once the DB assigns po.id.
    po = PurchaseOrder(
        po_id="__pending__",   # replaced immediately after flush
        sku_id=sku_id,
        qty_ordered=quantity,
        qty_received=0,
        status=POStatus.CONFIRMED,
        notes=notes,
        created_at=now,
        updated_at=now,
    )
    db.add(po)
    db.flush()  # write row + get auto-assigned po.id without committing yet

    # Build the human-readable po_id from the guaranteed-unique integer id
    po.po_id = f"PO-{now.year}-{po.id:06d}"
    _commit_and_refresh(db, po)

    result = po.to_dict()
    result["on_order_stock"] = po_on_order_stock(db, sku_id)
    return result


# ---------------------------------------------------------------------------
# Receive (full or partial)
# ---------------------------------------------------------------------------

def po_receive(db: Session, po_id: str, quantity_received: int) -> dict:
    """
    Record receipt of `quantity_received` units against an open PO.

    Business rules:
    - PO must be CONFIRMED or PARTIALLY_RECEIVED.
    - quantity_received must be 1 … remaining_qty (inclusive).
    - If quantity_received == remaining_qty → status becomes RECEIVED.
    - Otherwise → status becomes PARTIALLY_RECEIVED, qty_received accumulates.

    Raises:
        PONotFound            – po_id does not exist
        POInvalidTransition   – PO status does not allow receiving
        POInvalidQuantity     – quantity_received outside [1, remaining_qty]
    """
    po = _get_or_raise(db, po_id)

    if po.status not in POStatus.OPEN_STATUSES:
        raise POInvalidTransition(po_id, po.status, "receive stock")

    remaining = po.remaining_qty
    if quantity_received <= 0 or quantity_received > remaining:
        raise POInvalidQuantity(po_id, quantity_received, remaining)

    now = _now()
    po.qty_received += quantity_received
    po.updated_at = now
    po.status = POStatus.RECEIVED if po.qty_received >= po.qty_ordered else POStatus.PARTIALLY_RECEIVED

    _commit_and_refresh(db, po)

    result = po.to_dict()
    result["on_order_stock"] = po_on_order_stock(db, po.sku_id)
    return result


# ---------------------------------------------------------------------------
# Cancel
# ---------------------------------------------------------------------------

def po_cancel(db: Session, po_id: str) -> dict:
    """
    Cancel an open PO, removing its contribution to on_order_stock.

    Business rules:
    - Only CONFIRMED or PARTIALLY_RECEIVED POs can be cancelled.
    - Cancellation is irreversible.

    Raises:
        PONotFound            – po_id does not exist
        POInvalidTransition   – PO status does not allow cancellation
    """
    po = _get_or_raise(db, po_id)

    if po.status not in POStatus.OPEN_STATUSES:
        raise POInvalidTransition(po_id, po.status, "cancel")

    po.status = POStatus.CANCELLED
    po.updated_at = _now()
    _commit_and_refresh(db, po)

    result = po.to_dict()
    result["on_order_stock"] = po_on_order_stock(db, po.sku_id)
    return result


# ---------------------------------------------------------------------------
# Read
# ---------------------------------------------------------------------------

def po_get(db: Session, po_id: str) -> dict:
    """
    Return a single PO by po_id.

    Raises:
        PONotFound – po_id does not exist
    """
    po = _get_or_raise(db, po_id)
    result = po.to_dict()
    result["on_order_stock"] = po_on_order_stock(db, po.sku_id)
    return result


def po_list_by_sku(db: Session, sku_id: str) -> list[dict]:
    """
    Return all POs for a SKU, newest first.

    Returns a plain list of PO dicts. on_order_stock is a SKU-level aggregate,
    not a per-PO field — the caller is responsible for fetching it separately
    via po_on_order_stock() and placing it at the appropriate response level.
    """
    rows = (
        db.query(PurchaseOrder)
        .filter(PurchaseOrder.sku_id == sku_id)
        .order_by(PurchaseOrder.created_at.desc())
        .all()
    )
    return [po.to_dict() for po in rows]


# ---------------------------------------------------------------------------
# on_order_stock (live SQL aggregate — never cached, never computed in Python)
# ---------------------------------------------------------------------------

def po_on_order_stock(db: Session, sku_id: str) -> int:
    """
    Live SQL aggregate: SUM(qty_ordered - qty_received)
    for all CONFIRMED or PARTIALLY_RECEIVED POs for this SKU.

    Returns 0 if no open POs exist.

    Implemented as a single SQL expression — no Python-side row fetching.
    """
    result = (
        db.query(
            func.sum(PurchaseOrder.qty_ordered - PurchaseOrder.qty_received)
        )
        .filter(
            PurchaseOrder.sku_id == sku_id,
            PurchaseOrder.status.in_(POStatus.OPEN_STATUSES),
        )
        .scalar()
    )
    return result or 0


# ---------------------------------------------------------------------------
# Backward-compatibility shim
# ---------------------------------------------------------------------------
# Some callers (e.g. forecast_router) still import PurchaseOrderService by name.
# This thin wrapper delegates to the module functions so those callers continue
# to work without modification during a gradual migration.

class PurchaseOrderService:
    """
    Deprecated: use the module-level functions (po_create, po_receive, …)
    directly instead of this class wrapper.
    This shim will be removed once all callers have been updated.
    """
    create            = staticmethod(po_create)
    receive           = staticmethod(po_receive)
    cancel            = staticmethod(po_cancel)
    get               = staticmethod(po_get)
    list_by_sku       = staticmethod(po_list_by_sku)
    get_on_order_stock = staticmethod(po_on_order_stock)
