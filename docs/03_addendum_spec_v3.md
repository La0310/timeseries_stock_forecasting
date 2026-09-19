# Addendum: Constraints, Data Archetypes & Rules for Antigravity (v3)

Read alongside `01_product_spec.md` and `02_pipeline_workflow.md`. Treat every rule in this specification as an immutable hard constraint during implementation.

---

## 1. Feature, Target & Backtest Leakage Prevention (Hard Rule)

- Under no circumstances should `Demand Forecast` or `Units Ordered` be used as training inputs or inference features.
- In the retail inventory schema (`Date`, `Store ID`, `Product ID`, `Category`, `Region`, `Inventory Level`, `Units Sold`, `Units Ordered`, `Demand Forecast`, `Price`, `Discount`, `Weather Condition`, `Holiday/Promotion`, `Competitor Pricing`, `Seasonality`):
  - `Demand Forecast` correlates 0.997 with actual sales and constitutes synthetic target leakage.
  - `Inventory Level` is used strictly as an initial state for stock balance and replenishment calculations.
  - `Store ID` and `Region` exist for schema completeness only. Per the Demo Data Scope rule in `01_product_spec.md` §4, the demo provisions one store and one region per SKU; no calculation in this build aggregates across them.
- **New in v3**: the same principle extends to backtest scoring. No candidate model's backtest prediction may be computed by a function that has access to the actual values of the window it is being scored against. `02_pipeline_workflow.md` §2.2 enforces this structurally: there is no backtest-only prediction function; backtesting walks forward through history using the same generator production forecasting uses, one step ahead at a time, so a model cannot see the value it is about to be scored against. A model whose backtest score would only look good with access to its own answer is exactly the failure mode this rule exists to prevent, whether the access comes through a training feature or through a backtest function's argument list.

---

## 2. Benchmark Demo SKU Palette (`demo_inventory.csv`)

Antigravity must instantiate these 6 SKU archetypes so that dynamic model routing is visibly demonstrated across every quadrant of the ADI/CV² matrix, plus one seasonality overlay. `Expected Model` values are drawn exactly from `ModelRouter.MODEL_REGISTRY` in `02_pipeline_workflow.md`, so a real backtest can actually produce them; nothing here names a model the router isn't allowed to select.

| SKU ID | Product Name | Category | Pack Size | Lead Time (L) | Target Pattern | Expected Model | Primary Routing Driver |
|---|---|---|---|---|---|---|---|
| **SKU-101** | Coca-Cola 330ml Can | Beverage | 24 | 7d | Smooth | **Chronos-2** | ADI = 1.02 < 1.32, CV² = 0.18 < 0.49. High continuous turn, low seasonality overlay. |
| **SKU-202** | HP LaserJet Drum Unit | Office | 1 | 14d | Intermittent | **TSB** | ADI = 2.80 ≥ 1.32, CV² = 0.22 < 0.49. Infrequent sales, consistent order size. |
| **SKU-303** | DeWalt 18V Cordless Drill | Hardware | 4 | 21d | Erratic | **LightGBM** | ADI = 1.10 < 1.32, CV² = 0.85 ≥ 0.49. Daily transactions, promo-driven spikes. |
| **SKU-404** | Industrial 4-Inch Gate Valve | MRO / Parts | 1 | 30d | Lumpy | **SBA** | ADI = 3.40 ≥ 1.32, CV² = 1.12 ≥ 0.49. Sparse orders, erratic volumes; SBA is the registry's bias-corrected, Croston-family model built for this quadrant. |
| **SKU-505** | LED Fairy String Lights | Seasonal overlay | 12 | 14d | Smooth, with high quarterly seasonality | **SARIMAX** | ADI ≈ 1.05 < 1.32, CV² ≈ 0.24 < 0.49 (Smooth by the intermittency test), but lag-91 autocorrelation ≥ 0.6 (High, quarterly). SARIMAX must win this SKU's backtest on merit, because its generator models the seasonal component the others don't, not because of a hardcoded rule. |
| **SKU-606** | Multipurpose Copy Paper A4 | Office | 5 | 7d | High-Volume Smooth | **ETS** | ADI = 1.00 < 1.32, CV² = 0.08 < 0.49. Stable baseline staple. |

Note on SKU-505: "Seasonal" is not a fifth intermittency class, the classifier only ever returns Smooth / Intermittent / Erratic / Lumpy (§3.1). Seasonality is a second, independent signal computed alongside it. SKU-505's mock sales generator must embed a genuine quarterly/holiday demand spike large enough that the lag-91 autocorrelation test reads High and SARIMAX's backtest WAPE beats Chronos-2's as a result. The ADI/CV² figures above are illustrative targets for shaping that generator, not hardcoded output; the live values are still computed dynamically by `DemandClassifier.analyze()`, per the no-hardcoded-numbers rule in Directive 3 below.

All six SKUs seed `on_order_stock = 0` at initial demo load (no open purchase orders at t=0); Inventory Position therefore equals Current Stock until a user confirms a PO in-session (Screen 3), demonstrating the IP-based order gate in real time. See the worked example below.

**Note on limited-history coverage (clarification added):** all six archetypes above carry 273+ days of history by design, none of them is short. The limited-history fallback (`01_product_spec.md` Guardrail 3: fewer than 14 records, or every candidate model scoring `null`) is therefore never exercised by the default demo dataset alone. It is a real path in the system, not dead code, since a planner can always look up a SKU newly added to the catalog. It must still be validated, for example against a manually truncated slice of one archetype's own history, and the resulting banner and `limited_history` flag (`02_pipeline_workflow.md` §3.1) checked directly, rather than left to whichever archetype happens to be shortest.

### 2.1 Worked Example: Inventory-Position Gating

Uses SKU-101's own Screen 2 numbers (Current Stock $=40$, $SS=11$, $ROP=49$, 28-day demand $=152$, Pack Size $=24$), with one change: assume a PO for 60 units, placed last cycle, has not yet arrived (`on_order_stock = 60`).

| | Current Stock | On-Order | Inventory Position | ROP | Status | Order gate | Recommended Order |
|---|---|---|---|---|---|---|---|
| **v2 logic** (Current Stock only, no gate) | 40 | not tracked | not computed | 49 | Critical (40 ≤ 49) | $\max(0,\ 152+11-40)=123$ | 144 units (6 cases), re-ordered **on top of** the 60 units already inbound |
| **v3 logic** (IP-gated) | 40 | 60 | 100 | 49 | Healthy (100 > 1.5×49 = 73.5) | $IP(100) > ROP(49) \Rightarrow 0$ | 0 units, correctly |

This is the literal bug Revision Log item 1 (§6) closes. v2 had no way to see the 60 units already in the pipeline, so it would recommend ordering the same 144 units again every planning cycle until the PO physically arrived and raw Current Stock itself rose past ROP. v3 sees the on-order stock immediately, because Inventory Position, not raw on-hand stock, is what both `status` and the order gate are computed from.

---

## 3. Strict Mathematical Formulations

### 3.1 Demand Intermittency, Volatility & Seasonality

$$\text{Positive mask } m_t = \mathbb{I}(y_t > 0), \quad ADI = \frac{N}{\sum_{t=1}^N m_t}, \quad CV^2 = \left(\frac{\operatorname{std}(y_{t\mid m_t=1})}{\operatorname{mean}(y_{t\mid m_t=1})}\right)^2$$

- Cutoffs: $ADI_{\text{threshold}} = 1.32$, $CV^2_{\text{threshold}} = 0.49$. See §5 for a known limitation of these fixed cutoffs on real (non-demo) catalogs.

$$\rho_k = \frac{\sum_{t=1}^{N-k}(y_t-\bar y)(y_{t+k}-\bar y)}{\sum_{t=1}^{N}(y_t-\bar y)^2}, \quad k \in \{7, 91\}$$

- Only estimated when $N \ge 3k$; otherwise `null` ("Insufficient History"). Labeled High (≥0.6) / Medium (0.3-0.6) / Low (<0.3) by magnitude.

### 3.2 Inventory Position, Category-Differentiated Safety Stock & Reorder Trigger

**Category Service Levels:**

| Category | Target Service Level | Z-score | Rationale |
|---|---|---|---|
| Beverage | 95% | 1.65 | Fast-turning, short lead time; a short stockout is cheap to recover from. |
| Office | 95% | 1.65 | Fast-turning consumables; same reasoning as Beverage. |
| Hardware | 97.5% | 1.96 | Higher unit cost and promo-driven demand spikes raise the cost of an unfilled sale. |
| MRO / Parts | 99% | 2.33 | Long lead time, and a stockout can stall the buyer's own operations, not just lose a sale. |
| Seasonal overlay | 90% | 1.28 | Overstock and markdown risk after the season outweighs the risk of a short stockout during it; this is the one category where the tradeoff runs the other way. |

Any Category not listed above uses the 95% / Z = 1.65 default rather than failing.

$$SS = \left\lceil Z_{\text{category}} \times \sigma_{\text{daily}} \times \sqrt{L} \right\rceil, \qquad ROP = \sum_{t=1}^{L}\hat y_t + SS$$

**`σ_daily` definition (`daily_sales_std`, clarification added):** the sample standard deviation (`ddof=1`) of `Units Sold` across **all** N days in history, including zero-demand days. It is computed once by `DemandClassifier.analyze()` (`02_pipeline_workflow.md` §2.1) and returned as `demand_analysis.daily_sales_std`, alongside `cv_squared`. This is deliberately **not** the same statistic as `CV²`'s `σ_{y>0}`: `CV²` measures order-size volatility conditional on a sale occurring, and exists solely to classify intermittency (Smooth / Intermittent / Erratic / Lumpy). `daily_sales_std` measures unconditional day-to-day demand variability, the quantity that both the Safety Stock formula above and the P10/P90 forecast band (`02_pipeline_workflow.md` §2.2) are meant to buffer against. `forecast_engine.py` and `inventory_recommender.py` must consume `demand_analysis.daily_sales_std` as passed through by `forecast_router.py`; neither module recomputes it, and neither substitutes `cv_squared`'s intermediate non-zero std for it. This keeps the value computed exactly once, per the Single Source of Truth guardrail (`01_product_spec.md` §4 item 4).

**Inventory Position:**

$$IP = \text{OnHand} + \text{OnOrder}$$

A `Backorders` term was considered for this formula and dropped (§6, item 2): nothing in this build tracks backorders anywhere in the schema or the purchase-order flow, and a term with no data source behind it is worse than no term at all. If backorder tracking is added later, IP should be revisited then rather than approximated now.

- Status is derived from ROP and IP together, never a separate rule run in parallel:
  - IP ≤ ROP → **Critical**
  - ROP < IP ≤ 1.5·ROP → **Low**
  - IP > 1.5·ROP → **Healthy**

### 3.3 Packaging MOQ / Case Pack Rounding, Inventory-Position-Gated

$$Q_{\text{net}} = \begin{cases} 0 & IP > ROP \\ \max\left(0,\ \sum_{t=1}^{\text{horizon\_days}} \hat y_t + SS - IP\right) & IP \le ROP \end{cases}, \qquad Q_{\text{recommended}} = \left\lceil \frac{Q_{\text{net}}}{\text{Pack Size}} \right\rceil \times \text{Pack Size}$$

`horizon_days` defaults to 28. The gate is the fix for the "Healthy-but-still-orders" contradiction: once IP has recovered past ROP, the system stops asking for more, regardless of what the longer 28-day extrapolation alone would suggest. Only a SKU at or below ROP (Critical) computes a top-up quantity, and it tops up against the full review-horizon target, not just the lead-time gap. A "Low" SKU (between ROP and 1.5·ROP) is a warning state, not an ordering trigger; it becomes actionable only once it decays to Critical.

### 3.4 Model Score, with Zero-Guard

$$\text{Score}_m = \begin{cases} \max\left(0, (1 - \text{WAPE}_m) \times 100\right) & \sum_t y_t > 0 \text{ in the backtest window} \\ \text{null} & \sum_t y_t = 0 \text{ even at full history} \end{cases}$$

- Backtest window expands 14 → 28 → 56 → full history until $\sum_t y_t > 0$. A `null` score excludes that model from `max()` selection; if every candidate is `null`, `selected_model` is `null` and the UI shows the limited-history fallback banner.
- WAPE here is always computed by the leak-free walk-forward methodology in `02_pipeline_workflow.md` §2.2 (see §1 above).

---

## 4. Antigravity Implementation Directives

1. **Mock Compute Transparency**: the UI must state: `"Candidate evaluations derived from historical backtesting benchmarks; inference generated via modular demo engine."` Never simulate artificial latency exceeding 500ms for demo interactions, including the new `POST /api/purchase-order` call.
2. **Component Separation**: do not write monolithic script blocks. Maintain strict separation between `demand_classifier.py`, `model_router.py`, `forecast_engine.py`, `inventory_recommender.py`, and `purchase_order_service.py`. `forecast_engine.py` owns backtesting and quantile generation, per `02_pipeline_workflow.md` §2.2; it is not a stub. `purchase_order_service.py` owns the on-order ledger; `inventory_recommender.py` reads on-order stock from it but does not manage it.
3. **Data Immutability**: do not hardcode numbers into frontend components. All displayed figures, including `Status`, `Weekly Seasonality`, `Inventory Position`, `Service Level`, and the candidate model table, must be parsed dynamically from the API response payloads.
4. **Resilience**: fall back to a moving-average baseline and display an inline banner (`"Limited history: Defaulting to moving-average baseline."`) whenever either (a) fewer than 14 historical records exist, or (b) every candidate model scores `null` (§3.4). Unchanged by the Inventory Position work: these two conditions are about forecast confidence, not stock levels, and are independent of `status` or the order gate.
5. **Single Source of Truth**: any field with UI-facing semantics (Status badges, selected model, order quantities, on-order stock, `new_status`) is computed exactly once in a backend service and passed through unmodified. Frontend components render; they do not re-derive.
6. **Field-Name Contract**: frontend, API schema, and service return dicts use identical keys, no renaming across layers. This now also covers `on_order_stock`, `inventory_position`, `service_level_z`, and `new_status`.
7. **Backtest Methodology Constraint (New)**: `forecast_engine.py`'s backtest scoring must call the same per-model point-forecast generator used in production, at each expanding prefix of history, and compare it to the realized next value (§2.2 walk-forward, in `02_pipeline_workflow.md`). No model may have a second, backtest-only prediction function. This is enforced as an interface constraint, not a code-review checklist item, because a checklist is exactly how the leak this revision fixes got written in the first place.
8. **Category Service-Level Configuration (New)**: the `SERVICE_LEVEL_Z` mapping lives in one place in `inventory_recommender.py` (`02_pipeline_workflow.md` §2.4), not inlined at each call site, so the stockout-cost assumption behind every category's Safety Stock is auditable in one place. A Category absent from the mapping uses the documented 95% default instead of raising an error.
9. **Mock Model Naming (New, clarification added)**: the six `MODEL_REGISTRY` entries (`Chronos-2`, `LightGBM`, `SARIMAX`, `ETS`, `TSB`, `SBA`) are labels attached to the deterministic formulas specified in `02_pipeline_workflow.md` §2.2.1. They are named after real forecasting methods for narrative and UI flavor only. Do not `pip install` `chronos-forecasting` or `lightgbm`, do not load a pretrained checkpoint or call an external inference API for any of the six, and do not fit or train a real statistical or ML model at request time (including a real `statsmodels` SARIMAX/ETS fit). Every one of the six must resolve using only NumPy/pandas arithmetic, comfortably inside the <500ms latency ceiling in Directive 1.

---

## 5. Known Limitation: Threshold Generalization Beyond the Demo Catalog

The 1.32 / 0.49 ADI/CV² cutoffs in §3.1 are tuned to this build's six curated demo archetypes, one clean SKU per class, so that dynamic model routing is visibly demonstrable. Running `DemandClassifier.analyze()` unchanged against a larger, real SKU catalog pulled through the enterprise ETL pipeline (outside this demo's scope) produced a degenerate result: nearly the entire catalog classified as Erratic. Real point-of-sale data carries enough promotional and pricing noise that CV² clears 0.49 for almost every SKU, regardless of whether that SKU's underlying demand is actually volatile in a way that should change which model wins its backtest.

This is a real limitation of fixed global thresholds, not a bug in the formula itself, and it is not fixed in this revision. It is recorded here so it is not silently inherited by whoever takes this demo toward a production catalog. The likely fix is catalog-relative thresholds (e.g., ADI/CV² percentile cutoffs computed from the customer's own SKU base at onboarding time) rather than the two hardcoded constants this demo uses. That is a calibration exercise against real data, not a code change to `demand_classifier.py`, and stays out of scope until there is a real catalog to calibrate against.

---

## 6. Revision Log (v2 → v3)

1. **"Healthy-but-still-orders" contradiction closed for real.** v2 fixed the status-badge version of this bug (single ROP-relative band, no more dual-rule disagreement) but left a second copy of it in the order-quantity calculation, which was still driven by full 28-day demand against raw Current Stock, independent of Status entirely. A SKU could read "Healthy" and still return a positive `recommended_order`. v3 closes this by gating the order quantity itself: $Q_{net}=0$ whenever $IP>ROP$ (§3.3). See the worked example in §2.1.
2. **Inventory Position introduced.** `on_order_stock` is now tracked (`purchase_order_service.py`, `02_pipeline_workflow.md` §2.5) and $IP=\text{OnHand}+\text{OnOrder}$ replaces raw Current Stock as the input to both `classify_status()` and the order gate (§3.2). Without this, the system would re-recommend the same order every planning cycle even after a PO had already been placed; see §2.1. A `Backorders` term was considered for the IP formula and dropped: nothing in this build tracks backorders, and an undefined term is worse than an absent one.
3. **Purchase order confirmation is now a real, closed-loop flow.** Screen 2's "Create Purchase Order" button previously led nowhere in the spec. It now calls a new `POST /api/purchase-order` endpoint and navigates to a new Screen 3 (`01_product_spec.md` §3), whose `new_status` field is computed once on the backend from the post-PO Inventory Position and rendered verbatim by the frontend (new Guardrail 6, `01_product_spec.md` §4).
4. **Category-differentiated Safety Stock.** The Z-score behind Safety Stock is no longer a single global constant; it is looked up per SKU Category (§3.2), because a stockout on an industrial spare part and a stockout on a can of soda do not cost the business the same amount, and in the Seasonal category the cost actually runs the other way (overstock risk exceeds stockout risk).
5. **Backtest leakage closed at the interface level.** `ForecastEngine.run_backtest()` no longer hands any model a function that can see the actual values of the window it's being scored against, the vulnerability that would let a routing engine be gamed by a leak coefficient close to 1.0 regardless of demand pattern. It now performs walk-forward evaluation, reusing the same point-forecast generator each model uses in production (`02_pipeline_workflow.md` §2.2). There is no longer a separate backtest-only prediction function per model for a leak to hide inside (§1, §4 item 7).
6. **Real-data threshold generalization flagged, not fixed.** An ETL run against a real catalog degenerated to nearly 100% "Erratic" under the fixed 1.32/0.49 cutoffs. Recorded as a known limitation (§5) rather than patched here, since the right fix is catalog-relative calibration, which needs a real catalog to calibrate against.
7. **Two pre-implementation ambiguities closed (clarification pass).** `_model_point_forecast` (§2.2 in `02_pipeline_workflow.md`) now has a concrete deterministic formula per registry model, plus an explicit statement that all six are mock stand-ins rather than real library integrations (§2.2.1, Directive 9 above). `daily_sales_std` now has a pinned definition, computation site, and an explicit distinction from `CV²`'s non-zero-only std (§3.2 above), closing a subtle-mismatch risk between the demand-diagnosis panel and the replenishment panel.
8. **SKU-101 worked example corrected (patch, pre-build review).** The Screen 2/3 wireframes (`01_product_spec.md` §3) and the IP-gating worked example above (§2.1) previously paired ADI=1.02/CV²=0.18 with a Safety Stock of 48 units; those don't reconcile under the SS formula in §3.2, since a smooth ~5.4-units/day beverage's own daily_sales_std comes out near 2.5, not the ~11 that SS=48 (at Z=1.65, L=7) implies. Current Stock, Safety Stock, ROP, the 28-day demand figure, and every downstream number derived from them are corrected so the example is honestly reproducible from the stated formula (Current Stock 40, SS 11, ROP 49; the final recommended order and PO quantity are unaffected at 144 units / 6 cases). §2.1's v2-logic cell also had a copy-paste error, crediting "144 already inbound" where the setup specifies 60, now fixed. The `/api/forecast` JSON example in `02_pipeline_workflow.md` §3.1 is corrected to match, and gains the `limited_history` field it was missing. Flagged in the same pass: none of the six demo archetypes exercises the limited-history fallback path, noted in §2 above so it isn't silently left unverified.

v2's own revision log (status-badge single-basis rule, `cases_recommended` field rename, SBA reachability, seasonality as an independent signal, WAPE zero-guard, `forecast_engine.py` fully specified) remains in force and is not re-litigated here.

**Minor**: `on_order_stock` is demo-session-scoped (in-memory, `purchase_order_service.py`), not persisted the way `demo_inventory.csv` is; a production build would back it with a real purchase-order table.
