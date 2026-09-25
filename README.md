# AI Demand Forecasting & Dynamic Model Routing

> **Predictive stock management system** — diagnoses SKU-level demand patterns, runs walk-forward backtest tournaments across 6 ML models, generates probabilistic forecasts, and manages the full purchase order lifecycle.

---

## ✨ Features

- **Demand Classification** — Auto ADI / CV² diagnosis (Smooth, Erratic, Intermittent, Lumpy)
- **Dynamic Model Routing** — Walk-forward backtest ranks SBA, TSB, ETS, SARIMAX, LightGBM, Chronos-2 by WAPE; best model selected per SKU with explainability + tie-break rule
- **Probabilistic Forecasts** — P10 / P50 / P90 quantile forecasts over 28-day horizon
- **Smart Replenishment** — Safety stock, Reorder Point, case-pack rounding per category
- **Purchase Order Lifecycle** — Create / Partial Receive / Full Receive / Cancel, persisted in SQLite, race-free PO IDs
- **What-If Scenarios** — Lead time, demand spike, and order quantity simulators

---

## 🏗️ Project Structure

```
timeseries_stock_forecasting/
│
├── README.md
├── .gitignore
│
└── stock-management/
    ├── requirements.txt
    ├── backend/
    │   ├── main.py                    # FastAPI entry point + lifespan
    │   ├── exceptions.py              # Domain exceptions (PONotFound, etc.)
    │   ├── api/
    │   │   ├── forecast_router.py     #   POST /api/forecast
    │   │   ├── purchase_order_router.py #  PO CRUD endpoints
    │   │   └── scenario_router.py     #   POST /api/scenario
    │   ├── db/
    │   │   ├── database.py            #   SQLAlchemy engine & session
    │   │   ├── init_db.py             #   create_all() on startup
    │   │   └── models.py              #   PurchaseOrder, ModelCache ORM
    │   ├── services/
    │   │   ├── demand_classifier.py   #   ADI / CV² demand diagnosis
    │   │   ├── forecast_engine.py     #   6 models + walk-forward backtest
    │   │   ├── model_router.py        #   WAPE ranking + tie-break routing
    │   │   ├── inventory_recommender.py # Reorder logic & safety stock
    │   │   ├── purchase_order_service.py # PO state machine
    │   │   └── scenario_engine.py     #   What-if scenario math
    │   └── data/
    │       └── demo_inventory.csv     #   M5-like synthetic dataset (30 SKUs)
    ├── frontend/
    │   ├── index.html                 # Inventory dashboard
    │   ├── product_detail.html        # SKU deep-dive & forecast panel
    │   ├── po_confirmation.html       # Purchase order confirmation
    │   └── js/
    │       ├── inventory_table.js
    │       ├── forecast_card.js
    │       └── po_modal.js
    ├── scripts/
    │   ├── m5_pipeline.py             # Regenerate demo_inventory.csv
    │   └── generate_demo_data.py
    └── tests/
        └── test_helpers.py
```

---

## 🚀 Quick Start

### Step 1 — Clone the repository

```bash
git clone https://github.com/La0310/timeseries_stock_forecasting.git
```

### Step 2 — Navigate to the app folder

> ⚠️ **Important:** The app lives inside `stock-management/`, not the repo root.
> The folder name has a space, so use quotes when using the command line.

**Windows Command Prompt:**
```cmd
cd "C:\Users\ADMIN\Desktop\Stock prediction project\stock-management"
```

**Windows PowerShell:**
```powershell
cd "C:\Users\ADMIN\Desktop\Stock prediction project\stock-management"
```

**If you cloned fresh from GitHub** (no spaces in path):
```bash
cd timeseries_stock_forecasting/stock-management
```

### Step 3 — Create and activate a virtual environment

```cmd
python -m venv .venv
.venv\Scripts\activate
```

> After activation your prompt will show `(.venv)` at the start.

### Step 4 — Install dependencies

```cmd
pip install -r requirements.txt
```

### Step 5 — Run the server

```cmd
uvicorn backend.main:app --reload
```

> **Must be run from inside the `stock-management/` folder.**
> If you run it from the wrong folder you will see:
> `ModuleNotFoundError: No module named 'backend'`
> — fix: `cd` into `stock-management/` first, then run uvicorn.

### Step 6 — Open the app

Go to **http://127.0.0.1:8000** in your browser.

---

## ⚠️ Common Errors & Fixes

| Error | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: No module named 'backend'` | Running uvicorn from the wrong folder | `cd` into `stock-management/` first |
| `The system cannot find the path specified` | Path has spaces, no quotes used | Wrap the path in `"double quotes"` |
| `Address already in use` | A previous server is still running | Close the old terminal or use `--port 8001` |
| `No module named 'lightgbm'` | Dependencies not installed | Run `pip install -r requirements.txt` |

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/forecast` | Full demand → model routing → forecast → replenishment pipeline |
| `POST` | `/api/purchase-order` | Create a new PO |
| `GET`  | `/api/purchase-orders?sku_id=` | List all POs for a SKU |
| `GET`  | `/api/purchase-order/{po_id}` | Get a single PO |
| `POST` | `/api/purchase-order/{po_id}/receive` | Receive full or partial stock |
| `POST` | `/api/purchase-order/{po_id}/cancel` | Cancel an open PO |
| `POST` | `/api/scenario` | Run a what-if scenario |

Interactive API docs: **http://127.0.0.1:8000/docs**

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| **Backend** | Python 3.12, FastAPI, Uvicorn |
| **Database** | SQLite + SQLAlchemy ORM |
| **ML Models** | LightGBM, statsmodels (SARIMAX, ETS), custom (TSB, SBA) |
| **Data** | Pandas, NumPy, Faker (M5-like synthetic data) |
| **Frontend** | Vanilla HTML / CSS / JavaScript, Chart.js |

---

## 🔁 Regenerate Demo Data

If you want to regenerate `demo_inventory.csv` from scratch:

```cmd
cd "C:\Users\ADMIN\Desktop\Stock prediction project\stock-management"
python scripts/m5_pipeline.py
```

This creates 30 SKUs × 365 days of M5-like synthetic demand data.
