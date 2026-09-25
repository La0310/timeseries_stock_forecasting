# 📦 AI Inventory Management & Demand Forecasting System

> An end-to-end intelligent stock management system that automatically diagnoses product demand patterns, benchmarks multiple forecasting models, and generates data-driven replenishment recommendations — with a full purchase order lifecycle backed by SQLite.

**Live demo:** clone + run in 5 minutes → see [Quick Start](#-quick-start)

---

## 🎯 What Problem This Solves

Most inventory systems force planners to manually pick a forecasting method and tune reorder points by gut feel. This leads to:
- **Stockouts** on fast-moving items → lost sales
- **Overstock** on slow/intermittent items → tied-up cash
- **Wrong model choice** — a method that works for steady beverage sales is wrong for lumpy spare-parts demand

This system **automates that decision**: it classifies each product's demand pattern, runs a live backtest tournament across 6 forecasting models, and selects the winner per SKU — with full explainability.

---

## ✨ Key Features

### 🧠 Intelligent Model Selection
- Classifies each SKU's demand into **Smooth / Erratic / Intermittent / Lumpy** using ADI & CV² metrics (Syntetos-Boylan framework)
- Runs **walk-forward backtesting** on 6 models simultaneously
- Selects the best model by WAPE score, with a **tie-break rule** (simpler model preferred when scores are within 5%)
- Returns a plain-English **explanation** of why that model was chosen

### 📈 Probabilistic Forecasting
- Generates **28-day forecasts** with P10 / P50 / P90 confidence bands
- Not just a point estimate — planners see optimistic, expected, and pessimistic scenarios

### 📊 Smart Replenishment
- Computes **Safety Stock** (95% service level, Z=1.65)
- Computes **Reorder Point** = forecast × lead time + safety stock
- Rounds recommended order quantity to the nearest **case pack size**
- Status calibrated **per category** (Beverage, Office, Hardware, MRO)

### 📦 Purchase Order Lifecycle
- Full **Create → Partial Receive → Full Receive → Cancel** state machine
- POs persisted in **SQLite** with race-free ID generation (AUTOINCREMENT + post-flush formatting)
- `on_order_stock` is always a **live SQL aggregate** — never a stale cached value

### 🔬 What-If Scenario Simulator
- Adjust lead time, demand multiplier, or order quantity
- See instantly how inventory position and reorder point change

---

## 🤖 The 6 Forecasting Models

| Model | Type | Best for |
|---|---|---|
| **SBA** (Syntetos-Boylan Approximation) | Statistical formula | Intermittent demand |
| **TSB** (Teunter-Syntetos-Babai) | Exponential smoothing | Intermittent + obsolescence risk |
| **ETS** (Holt-Winters) | Statistical | Smooth demand with trend/seasonality |
| **SARIMAX** | Statistical | Seasonal patterns, AIC-selected order |
| **LightGBM** | Machine Learning | Erratic / complex patterns |
| **Chronos-2** | Weighted average | Lightweight baseline |

The system selects the winner **per SKU, per forecast run** — not one model for everything.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     FastAPI Backend                      │
│                                                         │
│  POST /api/forecast                                     │
│    └─► DemandClassifier   (ADI / CV² diagnosis)        │
│    └─► ForecastEngine     (6 models × walk-forward)    │
│    └─► ModelRouter        (WAPE ranking + tie-break)   │
│    └─► InventoryRecommender (ROP + safety stock)       │
│                                                         │
│  POST/GET /api/purchase-order/{...}                     │
│    └─► PurchaseOrderService  (state machine)           │
│    └─► SQLite via SQLAlchemy  (persistent ledger)      │
│                                                         │
│  POST /api/scenario                                     │
│    └─► ScenarioEngine     (what-if recalculation)      │
└─────────────────────────────────────────────────────────┘
          │ StaticFiles mount
┌─────────────────────────────────────────────────────────┐
│                  Vanilla JS Frontend                     │
│  index.html          Inventory dashboard (12 SKUs)      │
│  product_detail.html Forecast chart + model card        │
│  po_confirmation.html PO create / receive / cancel      │
└─────────────────────────────────────────────────────────┘
          │
┌─────────────────────────────────────────────────────────┐
│  SQLite  stock_management.db                            │
│  ├── purchase_orders  (PO ledger)                       │
│  └── model_cache      (pickled fitted models + WAPE)    │
└─────────────────────────────────────────────────────────┘
```

---

## 🛠️ Tech Stack

| Layer | Technology | Why |
|---|---|---|
| Backend API | Python 3.11, FastAPI | Fast async API, auto OpenAPI docs |
| Database | SQLite + SQLAlchemy ORM | Zero-config persistent storage |
| Forecasting | statsmodels, LightGBM | Industry-standard time series + ML |
| Data | Pandas, NumPy, Faker | M5-like synthetic data with realistic constraints |
| Frontend | Vanilla HTML/CSS/JS, Chart.js | No build step, runs anywhere |

---

## 📁 Project Structure

```
timeseries_stock_forecasting/
│
├── stock-management/
│   ├── backend/
│   │   ├── main.py                    # FastAPI app entry point
│   │   ├── exceptions.py              # Domain exceptions (PONotFound etc.)
│   │   ├── api/                       # HTTP routers (thin layer — no business logic)
│   │   ├── db/                        # SQLAlchemy models, session, init
│   │   └── services/                  # All business logic (no FastAPI imports)
│   │       ├── demand_classifier.py
│   │       ├── forecast_engine.py     # 6 models + walk-forward backtest
│   │       ├── model_router.py        # WAPE ranking + tie-break + explanation
│   │       ├── inventory_recommender.py
│   │       ├── purchase_order_service.py
│   │       └── scenario_engine.py
│   ├── frontend/                      # Vanilla HTML/CSS/JS
│   └── data/
│       └── demo_inventory.csv         # M5-like synthetic data (30 SKUs × 365 days)
│
├── scripts/
│   └── m5_pipeline.py                 # Regenerate training data
├── tests/
│   └── test_helpers.py
└── README.md
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.11 (the `.venv` is configured for 3.11 — do **not** use 3.12+)

### Step 1 — Clone

```bash
git clone https://github.com/La0310/timeseries_stock_forecasting.git
```

### Step 2 — Navigate to the app folder

> ⚠️ The app lives inside `stock-management/`, not the repo root. The folder path has spaces — use quotes.

```cmd
cd "C:\Users\ADMIN\Desktop\Stock prediction project\stock-management"
```

Or if you cloned fresh (no spaces):
```bash
cd timeseries_stock_forecasting/stock-management
```

### Step 3 — Create & activate a virtual environment

```cmd
python -m venv .venv
.venv\Scripts\activate
```

> Your prompt will show `(.venv)` when active.

### Step 4 — Install dependencies

```cmd
pip install -r requirements.txt
```

### Step 5 — Run the server

```cmd
uvicorn backend.main:app --reload
```

> **Must be run from inside `stock-management/`.** If you see `ModuleNotFoundError: No module named 'backend'`, you are in the wrong folder.

### Step 6 — Open the app

- **Dashboard:** http://127.0.0.1:8000
- **API docs:** http://127.0.0.1:8000/docs

> First load takes ~1–2 minutes — the system is fitting and backtesting 6 models per SKU. Results are cached in SQLite; subsequent loads are instant.

---

## ⚠️ Common Errors

| Error | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: No module named 'backend'` | Wrong working directory | `cd` into `stock-management/` first |
| `No module named 'pydantic_core._pydantic_core'` | `.pyd` binary built for wrong Python version | `pip install --force-reinstall pydantic-core pydantic` |
| `Unable to import required dependency numpy` | Same binary mismatch | `pip install --force-reinstall numpy pandas scipy scikit-learn lightgbm` |
| `The system cannot find the path specified` | Path has spaces, no quotes | Wrap path in `"double quotes"` |
| `Address already in use` | Old server still running | Close the old terminal, or use `--port 8001` |

---

## 📡 API Reference

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/forecast` | Full pipeline: demand diagnosis → model selection → 28-day forecast → replenishment |
| `POST` | `/api/purchase-order` | Create a new purchase order |
| `GET` | `/api/purchase-orders?sku_id=` | List all POs for a SKU |
| `GET` | `/api/purchase-order/{po_id}` | Get a single PO |
| `POST` | `/api/purchase-order/{po_id}/receive` | Record full or partial stock receipt |
| `POST` | `/api/purchase-order/{po_id}/cancel` | Cancel an open PO |
| `POST` | `/api/scenario` | Run a what-if scenario |

Full interactive docs: **http://127.0.0.1:8000/docs**

---

## 📊 Training Data

The system uses **M5-like synthetic data** generated by `scripts/m5_pipeline.py`:

- **30 SKUs** × **365 days** of daily sales history
- 5 demand archetypes: Smooth (Beverage), Intermittent (Office), Erratic (Hardware), Lumpy (MRO), Seasonal
- Non-demand fields (lead time, pack size, price, supplier) generated by **Faker** following real business constraints

To regenerate:
```cmd
python scripts/m5_pipeline.py
```

---

## 🔑 Design Decisions

| Decision | Rationale |
|---|---|
| Module functions, not a service class | No shared state — a class was just a namespace wrapper |
| `db.flush()` before formatting PO ID | Gets the DB-assigned `AUTOINCREMENT` id atomically — eliminates `COUNT(*)+1` race condition |
| SQL `SUM()` for on_order_stock | One query, always consistent — never stale from in-memory accumulation |
| `threading.Lock` around LightGBM | LightGBM's C library is not thread-safe on Windows; concurrent calls cause access violations |
| Domain exceptions, not `HTTPException` in services | Keeps the service HTTP-agnostic — usable from CLI, tests, or background jobs |
| Lazy model cache in SQLite | Avoids re-fitting on every API call; cache invalidated by history hash (md5) |
