"""
database.py
SQLAlchemy engine, session factory, and declarative Base.

The DB file lives at backend/data/stock_management.db so all persistent
state (POs, model cache) is co-located with the inventory CSV.

Design principle: nothing outside this module imports `engine` directly.
All callers use `get_db()` (a FastAPI dependency) or `SessionLocal()` in
scripts. Swapping to PostgreSQL is a one-line DATABASE_URL change here.
"""

from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker

# ---------------------------------------------------------------------------
# DB file path — relative to this file's location
# ---------------------------------------------------------------------------
_DB_DIR = Path(__file__).parent.parent / "data"
_DB_DIR.mkdir(parents=True, exist_ok=True)

DATABASE_URL = f"sqlite:///{_DB_DIR / 'stock_management.db'}"

# check_same_thread=False is required for SQLite when used with FastAPI's
# async request handling (multiple threads per process).
engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False},
    echo=False,  # set True locally to see all SQL statements
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    """Shared declarative base — all ORM models inherit from this."""
    pass


# ---------------------------------------------------------------------------
# FastAPI dependency
# ---------------------------------------------------------------------------

def get_db():
    """
    Yields a SQLAlchemy session and guarantees it is closed afterwards.
    Use as a FastAPI Depends():

        @router.get("/example")
        def example(db: Session = Depends(get_db)):
            ...
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
