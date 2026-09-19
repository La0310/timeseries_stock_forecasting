# Product Specification: AI Demand Forecasting & Dynamic Model Routing (v3)

## 1. Executive Summary & Objective

- **Product Name**: AI Demand Forecasting & Dynamic Model Routing module (embedded inside Core Stock Management).
- **Core Value Proposition**: Transform static inventory tracking into an automated, explainable replenishment workflow. Planners diagnose demand patterns, inspect why specific statistical or machine-learning models were selected, review probabilistic forecasts, convert recommendations directly into purchase orders, and see the resulting on-order stock reflected immediately in the SKU's inventory position and status.
- **Workflow Scope**: Stock Management → Demand Diagnosis (ADI / CV²) → Dynamic Model Routing → Probabilistic Forecast (P10, P50, P90) → Inventory-Position-Gated Replenishment (Safety Stock, Lead Time, Case Pack rounding) → Purchase Order Confirmation & On-Order Tracking.

---

## 2. System Architecture & Directory Layout

```text
stock-management/
├── frontend/
│   ├── index.html                    # Core inventory management view
│   ├── product_detail.html           # SKU deep dive & forecasting panel
│   ├── po_confirmation.html          # Screen 3: purchase order confirmation
│   └── js/
│       ├── inventory_table.js        # Stock table rendering & health badges
│       ├── forecast_card.js          # Reactive forecast generation & charts
│       └── po_modal.js               # PO payload builder & Screen 3 render
├── backend/
│   ├── main.py                       # Application entry point & routing
│   ├── api/
│   │   ├── forecast_router.py        # POST /api/forecast
│   │   └── purchase_order_router.py  # POST /api/purchase-order
│   ├── services/
│   │   ├── demand_classifier.py      # Syntetos-Boylan ADI & CV² + seasonality
│   │   ├── model_router.py           # WAPE-based model ranking & justification
│   │   ├── forecast_engine.py        # Walk-forward backtesting, quantile generation
│   │   ├── inventory_recommender.py  # Inventory Position, ROP, Safety Stock, Status
│   │   └── purchase_order_service.py # On-order ledger (on_order_stock per SKU)
│   └── data/
│       └── demo_inventory.csv        # Curated 6-archetype demo dataset
└── requirements.txt
```

`purchase_order_service.py` is new in v3. It owns the on-order ledger; `inventory_recommender.py` reads from it but does not manage it (component separation, `03_addendum_spec.md` §4 item 2).

---

## 3. Detailed User Experience & Screen Flow

### Screen 1: Inventory Table (Existing Core View)

Displays all managed SKUs with key inventory health metrics.

**Columns:**

- Product ID / SKU Name
- Category
- Current Stock (derived from `Inventory Level`)
- Sales / Week: the **trailing 7-day sum** of `Units Sold` (not a daily average). Displayed as an integer unit count, labeled `/wk`.
- Lead Time (Days): category-specific supplier turnaround
- Status: health badge, defined below

**Status badge (single-basis rule):**

Status is computed from exactly one reference point, the Reorder Point (ROP = lead-time demand + safety stock), compared against **Inventory Position (IP = OnHand + OnOrder)**, not raw on-hand stock alone. Comparing ROP to on-hand stock only, while a separate order-quantity calculation looked at full-horizon demand instead, is exactly how v2 could show a SKU as "Healthy" while still recommending an order for it (`03_addendum_spec.md` §6, item 1).

| Status | Condition |
|---|---|
| **Critical** | Inventory Position ≤ ROP |
| **Low** | ROP < Inventory Position ≤ 1.5 × ROP |
| **Healthy** | Inventory Position > 1.5 × ROP |

This band is continuous, there is no gap between Low and Healthy the way a separate "2× lead-time demand" rule would create.

**Single Source of Truth**: `Status` is computed exactly once, by `InventoryRecommender.classify_status()` in the backend (see `02_pipeline_workflow.md`), and returned as `replenishment.status` in the `/api/forecast` payload. `inventory_table.js` must render this value directly. It must never recompute Status client-side, and it must never substitute raw on-hand Current Stock for Inventory Position when reasoning about a badge, that substitution is what produced the original contradiction.

**Interaction:** Clicking any SKU row navigates to Screen 2.

---

### Screen 2: Product Detail & AI Replenishment View

Presents operational SKU details alongside the AI Demand Forecasting & Dynamic Model Routing card.

```text
+------------------------------------------------------------------------------------+
| SKU-101: Coca-Cola 330ml Can | Category: Beverage | Pack Size: 24 | Lead Time: 7d   |
| Current Stock: 40 units      | Safety Stock Target: 11 units | Status: Critical    |
+------------------------------------------------------------------------------------+
| [ Generate AI Forecast ]                                                            |
+------------------------------------------------------------------------------------+
| 1. DEMAND PATTERN DIAGNOSIS                                                         |
| Pattern: Smooth Demand                                                              |
| ADI: 1.02 (< 1.32) | CV²: 0.18 (< 0.49) | Zero Ratio: 2%                           |
| Weekly Seasonality (ACF-7): 0.12 - Low | Quarterly Seasonality (ACF-91): n/a        |
+------------------------------------------------------------------------------------+
| 2. MODEL ROUTING & BENCHMARK SCORE                                                  |
| Selected Model: Chronos-2 (Score: 91.7%)                                            |
| Why this model?                                                                     |
| - Continuous non-zero sales frequency (ADI 1.02)                                    |
| - Low transaction variance, no material seasonality overlay                         |
| - Outperformed classical baselines in a leak-free walk-forward backtest             |
|                                                                                      |
| Candidate Model Evaluation:                                                         |
| Model         | Backtest Accuracy (100 - WAPE) | Selection Status                   |
| Chronos-2     | 91.7%                          | Selected                           |
| LightGBM      | 88.4%                          | Candidate                          |
| SARIMAX       | 82.3%                          | Candidate                          |
| ETS           | 79.1%                          | Candidate                          |
| TSB           | 64.0%                          | Rejected (non-intermittent)        |
| SBA           | 58.2%                          | Rejected (non-lumpy)               |
+------------------------------------------------------------------------------------+
| 3. PROBABILISTIC DEMAND PROJECTION                                                  |
| Next 7 Days: 38 units | Next 14 Days: 76 units | Next 28 Days: 152 units            |
| (Interactive Plotly Chart: Actuals + P10/P50/P90 Forecast Ribbon)                   |
+------------------------------------------------------------------------------------+
| 4. REPLENISHMENT RECOMMENDATION                                                     |
| Current On-Hand Stock:             40 units                                        |
| On-Order Stock:                     0 units                                        |
| Inventory Position (IP):           40 units   (OnHand + OnOrder)                   |
| Expected Lead-Time Demand (7d):    38 units                                        |
| Service Level:                 95% (Z=1.65, Beverage)                              |
| Required Safety Buffer (SS):       11 units                                        |
| Reorder Point (ROP):               49 units                                        |
| Status:                       Critical (IP ≤ ROP)                                  |
| Target Review Cycle Demand (28d):  152 units                                       |
| Unrounded Reorder Need:            123 units   (IP ≤ ROP, gate is open)            |
| Pack Size Multiplier:               24 units / Case                                |
| ------------------------------------------------------------------------------------|
| FINAL RECOMMENDED ORDER: 144 units (6 Cases)                                        |
| [ Create Purchase Order ]                                                           |
+------------------------------------------------------------------------------------+
```

Every number on this screen (including `Status`, `Weekly Seasonality`, and the six-row candidate table) is parsed directly from the `/api/forecast` response. None of it is hardcoded in `forecast_card.js`.

The seasonality fields are real, computed values (autocorrelation at lag 7 and lag 91; see `02_pipeline_workflow.md` §2.1), not a placeholder label. Where history is too short to estimate a lag reliably, the field reads `n/a` rather than guessing.

The "On-Order Stock", "Inventory Position (IP)", and "Service Level" rows are new in v3. At initial demo load every SKU's on-order stock is 0, so IP equals Current Stock and the numbers above are unchanged from v2's worked example. They diverge only after a purchase order is confirmed in this session (Screen 3) or in the cross-cycle worked example in `03_addendum_spec.md` §2.1.

---

### Screen 3: Purchase Order Confirmation (New in v3)

Reached only via "Create Purchase Order" on Screen 2. Calls `POST /api/purchase-order` (`02_pipeline_workflow.md` §3.2) and renders exactly what that endpoint returns.

```text
+------------------------------------------------------------------------------------+
| Screen 3: Purchase Order Confirmation - SKU-101                                    |
+------------------------------------------------------------------------------------+
| PO Reference: PO-2026-000001                                                        |
| Quantity Ordered: 144 units (6 Cases)                                               |
| Expected Delivery: 7 days (Lead Time)                                               |
+------------------------------------------------------------------------------------+
| Updated Inventory Position                                                          |
| On-Hand Stock:            40 units                                                  |
| + On-Order Stock:        144 units   (this PO, newly recorded)                      |
| = Inventory Position:    184 units                                                  |
| Reorder Point (ROP):      49 units                                                  |
| Status:                Healthy      <- rendered verbatim from `new_status`          |
+------------------------------------------------------------------------------------+
| [ Return to Inventory Table ]                                                       |
+------------------------------------------------------------------------------------+
```

Every field here, including `new_status`, is parsed from the endpoint response. Per Guardrail 6 below, the frontend never infers or hardcodes this transition (e.g. assuming Critical always becomes Healthy). A partial fill, a second concurrent PO, or any other change to on-order stock could leave the SKU at Low instead, and only the backend's single Inventory Position calculation knows which actually happened.

---

## 4. Operational Guardrails

1. **No Invisible Black Boxes**: every forecast output must display its governing parameters (ADI, CV², seasonality ACF, backtest WAPE score, lead-time horizon L, and the per-category Service Level / Z-score behind Safety Stock, see `03_addendum_spec.md` §3.2).
2. **Actionable Units**: raw mathematical order quantities (e.g., 138 loose cans) must always be converted into supplier-acceptable packaging increments (e.g., 6 cases of 24) before presenting the purchase order trigger.
3. **Strict UI Feedback States**: display explicit skeleton loaders while calculating metrics ("Analyzing demand patterns and evaluating candidate models..."). Show the limited-history banner ("Limited history: Defaulting to moving-average baseline.") whenever either (a) fewer than 14 historical records exist, or (b) every candidate model returns an Insufficient Data score. Both are the same user-facing state: not enough signal to trust the routing decision. `POST /api/purchase-order` gets the same "Recording purchase order..." loading treatment under the same <500ms mock-latency rule (`03_addendum_spec.md` §4 item 1).
4. **Single Source of Truth**: any field with UI-facing semantics (Status badges, selected model, order quantities, on-order stock, `new_status`) is computed exactly once in a backend service and passed through unmodified. Frontend components render; they do not re-derive.
5. **Demo Data Scope**: `demo_inventory.csv` provisions exactly one `Store ID` and one `Region` per archetype SKU; no calculation in this build aggregates across them. `on_order_stock` is demo-session-scoped, held in `purchase_order_service.py`'s in-memory ledger, and resets when the backend restarts. It is not persisted the way `demo_inventory.csv` is.
6. **No Frontend Status Recomputation After a Stateful Action (New in v3)**: once a Purchase Order is confirmed, the frontend renders `new_status` from the `/api/purchase-order` response exactly as returned. It never assumes a specific before/after transition, and it never recomputes status from on-hand stock alone, doing so would silently ignore the on-order stock the PO itself just created and reintroduce the "Healthy-but-still-orders" failure mode this revision closed (`03_addendum_spec.md` §6, item 1).
