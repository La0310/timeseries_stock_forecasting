"""
models.py
SQLAlchemy ORM table definitions.

Tables
------
purchase_orders : Full PO lifecycle ledger.
model_cache     : Lazy-fitted model blobs + WAPE scores (used in Phase 2).

Design notes
------------
- `id` is an AUTOINCREMENT surrogate key. It is the source of truth for
  sequencing and is assigned atomically by the DB engine — eliminating the
  race condition that existed when using COUNT(*)+1 in application code.
- `po_id` is a formatted string ("PO-YYYY-NNNNNN") derived from `id` after
  insert. It has a UNIQUE constraint and is the public-facing identifier.
- `on_order_stock` is NOT stored as a column — it is always computed live as
  SUM(qty_ordered - qty_received) WHERE status IN ('CONFIRMED','PARTIALLY_RECEIVED').
  This prevents stale derived-value bugs.
- `model_cache` is created now so the schema is stable; it is populated in
  Phase 2 when real ML model fitting is added.
"""

from datetime import datetime, timezone

from sqlalchemy import Column, DateTime, Float, Index, Integer, LargeBinary, String, Text, UniqueConstraint

from backend.db.database import Base


def _now() -> datetime:
    """UTC-aware timestamp for default values."""
    return datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# PO status constants — import from here, never use raw strings in callers
# ---------------------------------------------------------------------------

class POStatus:
    CONFIRMED           = "CONFIRMED"
    PARTIALLY_RECEIVED  = "PARTIALLY_RECEIVED"
    RECEIVED            = "RECEIVED"
    CANCELLED           = "CANCELLED"

    # Statuses that still contribute to on_order_stock
    OPEN_STATUSES = (CONFIRMED, PARTIALLY_RECEIVED)


# ---------------------------------------------------------------------------
# purchase_orders
# ---------------------------------------------------------------------------

class PurchaseOrder(Base):
    __tablename__ = "purchase_orders"

    # Surrogate integer key — assigned atomically by the DB (race-free sequencing)
    id              = Column(Integer, primary_key=True, autoincrement=True)

    # Public-facing identifier formatted from `id` after insert ("PO-YYYY-NNNNNN")
    po_id           = Column(String, unique=True, nullable=False, index=True)

    sku_id          = Column(String, nullable=False, index=True)
    qty_ordered     = Column(Integer, nullable=False)
    qty_received    = Column(Integer, nullable=False, default=0)
    status          = Column(String, nullable=False, default=POStatus.CONFIRMED)
    notes           = Column(Text, nullable=True)
    created_at      = Column(DateTime(timezone=True), nullable=False, default=_now)
    updated_at      = Column(DateTime(timezone=True), nullable=False, default=_now, onupdate=_now)

    # ------------------------------------------------------------------
    # Computed helpers (not stored — derived on the fly from columns)
    # ------------------------------------------------------------------

    @property
    def remaining_qty(self) -> int:
        """Units still on order (not yet received)."""
        return self.qty_ordered - self.qty_received

    def to_dict(self) -> dict:
        """Serialise to a JSON-safe dict for API responses."""
        return {
            "po_id":            self.po_id,
            "sku_id":           self.sku_id,
            "qty_ordered":      self.qty_ordered,
            "qty_received":     self.qty_received,
            "remaining_qty":    self.remaining_qty,
            "status":           self.status,
            "notes":            self.notes,
            "created_at":       self.created_at.isoformat() if self.created_at else None,
            "updated_at":       self.updated_at.isoformat() if self.updated_at else None,
        }


# ---------------------------------------------------------------------------
# model_cache  (populated in Phase 2 — created now for schema stability)
# ---------------------------------------------------------------------------

class ModelCache(Base):
    __tablename__ = "model_cache"

    # cache_key = "{sku_id}|{model_name}|{history_hash}"
    cache_key       = Column(String, primary_key=True)
    model_blob      = Column(LargeBinary, nullable=False)   # pickled fitted model
    wape_score      = Column(Float, nullable=True)          # backtest WAPE at fit time
    history_hash    = Column(String, nullable=False)        # md5 of training series bytes
    fitted_at       = Column(DateTime(timezone=True), nullable=False, default=_now)
