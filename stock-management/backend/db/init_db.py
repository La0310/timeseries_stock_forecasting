"""
init_db.py
Creates all SQLAlchemy tables if they don't already exist.

Called once at application startup via the FastAPI lifespan context manager
in main.py. Safe to call repeatedly — `create_all` is idempotent (it skips
tables that already exist).

Running directly:
    python -m backend.db.init_db
"""

from backend.db.database import Base, engine

# Import all models so their metadata is registered with Base before
# create_all() is called. This import must stay even if the symbols
# are unused in this module.
import backend.db.models  # noqa: F401


def init_db() -> None:
    """Create all tables. Idempotent — safe to call on every startup."""
    Base.metadata.create_all(bind=engine)
    print("[init_db] Database tables verified / created.")


if __name__ == "__main__":
    init_db()
