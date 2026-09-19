# AI Demand Forecasting & Dynamic Model Routing

> **Predictive stock management system** — diagnoses SKU-level demand patterns, runs walk-forward backtest tournaments across statistical models, generates probabilistic forecasts, and produces inventory-position-gated purchase orders.

---

## ✨ Features

- **Demand Classification** — Automatic ADI / CV² diagnosis using the Syntetos-Boylan framework (Smooth, Erratic, Intermittent, Lumpy) plus seasonality detection
- **Dynamic Model Routing** — Walk-forward backtesting ranks SMA, EMA, Croston, and Holt-Winters by WAPE; best model is selected per SKU with full explainability
- **Probabilistic Forecasts** — Generates P10 / P50 / P90 quantile forecasts with confidence intervals
- **Smart Replenishment** — Inventory-position-gated recommendations with Safety Stock, Reorder Point, and Case Pack rounding
- **Purchase Order Workflow** — One-click PO generation with on-order tracking that updates inventory status in real time
- **What-If Scenarios** — Price/promotion scenario simulator showing baseline vs. modified demand side-by-side

---

## 🏗️ Project Structure

```
timeseries_stock_forecasting/
│
├── README.md
├── .gitignore
│
├── docs/                              # Design & specification documents
│   ├── 01_product_spec_v3.md          #   Product specification
│   ├── 02_pipeline_workflow_v3.md     #   Pipeline & API workflow
│   ├── 03_addendum_spec_v3.md         #   Addendum (scenarios, PO tracking)
│   └── 04_patch_notes_v3.1.md         #   Patch notes v3.1
│
├── stock-management/
│   ├── requirements.txt
│   ├── backend/
│   │   ├── main.py                    # FastAPI entry point
│   │   ├── api/
│   │   │   ├── forecast_router.py     #   POST /api/forecast
│   │   │   ├── purchase_order_router.py   POST /api/purchase-order
│   │   │   └── scenario_router.py     #   POST /api/scenario
│   │   ├── services/
│   │   │   ├── demand_classifier.py   #   ADI / CV² demand diagnosis
│   │   │   ├── model_router.py        #   WAPE-based model ranking
│   │   │   ├── forecast_engine.py     #   Walk-forward backtesting
│   │   │   ├── inventory_recommender.py   Reorder logic & safety stock
│   │   │   ├── purchase_order_service.py  On-order ledger
│   │   │   └── scenario_engine.py     #   What-if scenario math
│   │   └── data/
│   │       ├── demo_inventory.csv     #   6-archetype demo dataset
│   │       └── generate_demo_data.py  #   Script to regenerate demo data
│   └── frontend/
│       ├── index.html                 # Inventory dashboard
│       ├── product_detail.html        # SKU deep-dive & forecast panel
│       ├── po_confirmation.html       # Purchase order confirmation
│       └── js/
│           ├── inventory_table.js     #   Table rendering & health badges
│           ├── forecast_card.js       #   Forecast charts & model cards
│           └── po_modal.js            #   PO builder & confirmation flow
│
├── tests/
│   └── test_classifier.py            # Demand classifier unit tests
│
└── scripts/
    ├── validate.py                    # End-to-end pipeline validation
    ├── check_scenario_columns.py      # Debug: inspect scenario data columns
    └── diagnose_generators.py         # Debug: verify data generators
```

---

## 🚀 Quick Start

### Prerequisites

- **Python 3.10+**

### Installation

```bash
# Clone the repository
git clone https://github.com/La0310/timeseries_stock_forecasting.git
cd timeseries_stock_forecasting

# Create a virtual environment
python -m venv .venv

# Activate it
# Windows:
.venv\Scripts\activate
# macOS / Linux:
source .venv/bin/activate

# Install dependencies
pip install -r stock-management/requirements.txt
```

### Run the Application

```bash
cd stock-management
uvicorn backend.main:app --reload
```

Open **http://127.0.0.1:8000** in your browser.

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/forecast` | Run the full demand-diagnosis → model-routing → forecast → replenishment pipeline for a given SKU |
| `POST` | `/api/purchase-order` | Record a purchase order, recompute inventory position and status |
| `POST` | `/api/scenario` | Run a what-if price/promotion scenario comparing baseline vs. modified demand |

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| **Backend** | Python, FastAPI, Uvicorn |
| **Data** | Pandas, NumPy, SciPy |
| **Frontend** | Vanilla HTML / CSS / JavaScript |
| **Charts** | Chart.js (CDN) |

---

## 📂 Documentation

Detailed design specifications are in the [`docs/`](docs/) directory:

- [Product Specification v3](docs/01_product_spec_v3.md) — Full feature spec and screen flows
- [Pipeline & Workflow v3](docs/02_pipeline_workflow_v3.md) — API schemas and algorithm details
- [Addendum v3](docs/03_addendum_spec_v3.md) — Scenario simulator and PO tracking additions
- [Patch Notes v3.1](docs/04_patch_notes_v3.1.md) — Latest changes and fixes

---

## 📋 Scripts & Tests

```bash
# Run the demand classifier tests
python -m pytest tests/

# Validate the full pipeline end-to-end
python scripts/validate.py
```
