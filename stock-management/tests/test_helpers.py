"""
test_helpers.py
Test-only utilities — NEVER import this from production code.

Contains:
  reset_all(db) — wipe all PO and model-cache rows for test isolation.
"""

from __future__ import annotations

import os

from sqlalchemy.orm import Session

from backend.db.models import ModelCache, PurchaseOrder

_TEST_ENVS = {"test", "testing", "pytest"}


def reset_all(db: Session) -> None:
    """
    Drop all PurchaseOrder and ModelCache rows.

    Safety guard: raises RuntimeError if called outside a test environment.
    Set the APP_ENV environment variable to 'test' or 'testing' to enable.
    """
    env = os.getenv("APP_ENV", "").lower()
    if env not in _TEST_ENVS:
        raise RuntimeError(
            f"reset_all() is only allowed when APP_ENV is one of {_TEST_ENVS}. "
            f"Current APP_ENV={env!r}. Never call this in production."
        )
    db.query(ModelCache).delete()
    db.query(PurchaseOrder).delete()
    db.commit()
