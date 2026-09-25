"""
main.py
FastAPI application entry point.
Mounts forecast and purchase-order routers, serves frontend static files,
enables CORS for local development.

Spec: 01_product_spec_v3.md §2 (directory layout)
"""

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
# pyrefly: ignore [missing-import]
from fastapi.staticfiles import StaticFiles

from backend.api.forecast_router import router as forecast_router
from backend.api.purchase_order_router import router as po_router
from backend.api.scenario_router import router as scenario_router
from backend.db.init_db import init_db


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan handler.
    - Startup: initialise the SQLite database (idempotent create_all).
    - Shutdown: nothing to do (SQLAlchemy closes connections automatically).
    """
    init_db()
    yield


app = FastAPI(
    title="AI Demand Forecasting & Dynamic Model Routing",
    description=(
        "Demand diagnosis (ADI/CV²), walk-forward backtest model routing, "
        "probabilistic P10/P50/P90 forecasts, and IP-gated replenishment."
    ),
    version="3.2.0",
    lifespan=lifespan,
)

# CORS — allow all origins for local demo development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# API routers
app.include_router(forecast_router)
app.include_router(po_router)
app.include_router(scenario_router)

# Serve frontend static files at root
_FRONTEND = Path(__file__).parent.parent / "frontend"
app.mount("/", StaticFiles(directory=str(_FRONTEND), html=True), name="frontend")
