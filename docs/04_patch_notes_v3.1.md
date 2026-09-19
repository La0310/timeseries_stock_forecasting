# Patch v3.1 — Pre-Build Correction Pass

Applied directly to `01_product_spec_v3.md`, `02_pipeline_workflow_v3.md`, and `03_addendum_spec_v3.md` (already reflected in the copies you have). This document is the audit trail: what was wrong, why, and exactly what changed. Logged as Revision Log item 8 in `03` §6.

---

## Patch 1 (Critical): SKU-101's worked example didn't reconcile with its own formula

**Root cause.** The Screen 2/3 wireframes and the `03` §2.1 worked example paired ADI=1.02 / CV²=0.18 (a smooth, ~5.4-unit/day beverage) with a Safety Stock of 48 units. Working the SS formula backward (`SS = ceil(Z × σ_daily × √L)`, Z=1.65, L=7), SS=48 requires `daily_sales_std ≈ 11`. But a smooth series at that mean and CV² actually produces `daily_sales_std ≈ 2.4–2.5` (day counts at modest volume behave close to Poisson, where std ≈ √mean). The gap is roughly 4-5x, not a rounding difference. Every number downstream of SS (ROP, the order-gate arithmetic, the post-PO Inventory Position) inherited the error.

**What changed** (all four now reconcile under the stated formula, with the final recommended order and PO quantity unaffected):

| Field | Old | New | Where |
|---|---|---|---|
| Current Stock | 84 | 40 | `01` Screen 2 header, Screen 2 §4, Screen 3 |
| Zero Ratio | 0% | 2% | `01` Screen 2 §1 (ADI=1.02 implies ~2% zero-demand days, not exactly 0) |
| Next 14 Days | 81 | 76 | `01` Screen 2 §3 (linear scaling of the 7-day figure; smooth, non-seasonal) |
| Next 28 Days | 174 | 152 | `01` Screen 2 §3 and §4 |
| Safety Stock (SS) | 48 | 11 | `01` Screen 2 §4 |
| Reorder Point (ROP) | 86 | 49 | `01` Screen 2 §4, Screen 3 |
| Unrounded Reorder Need | 138 | 123 | `01` Screen 2 §4 |
| Inventory Position after PO | 228 | 184 | `01` Screen 3 |
| **Final Recommended Order / PO qty** | 144 (6 cases) | **144 (6 cases), unchanged** | carries through unaffected |

Same corrections carried into `03` §2.1's worked example (Current Stock 40, SS 11, ROP 49, 28-day demand 152) and into `02` §3.1/§3.2's JSON examples (`daily_sales_std`: 0.3 → 2.5, `zero_demand_ratio`: 0.0 → 0.02, `safety_stock`: 48 → 11, `reorder_point`: 86 → 49, `current_stock`/`inventory_position`: 84 → 40, post-PO `inventory_position`: 228 → 184). The P10/P90 quantile example was recomputed honestly from `daily_sales_std=2.5`; it now widens fast and clamps P10 to 0 by day 2, which is the formula behaving correctly at modest per-day volume, not a new error, and `02` §3.1 now says so explicitly.

**Bonus find, same review pass:** `03` §2.1's v2-logic table cell said the recommended order was placed "on top of the 144 already inbound," but the scenario's own setup specifies 60 units already inbound. Copy-paste error, now reads "60 units already inbound."

---

## Patch 2: `limited_history` given a home in the schema

The implementation plan (correctly) wants a `limited_history` flag so the frontend never has to infer the fallback-banner condition itself, but `02` §3.1's frozen response schema never defined one. **Fixed**: `limited_history` (boolean) is now a top-level field in the `/api/forecast` response, next to `sku_id`, with a one-line definition in the prose immediately below the JSON block (set whenever `len(history) < 14` or every candidate model scores `null`, per `03` §4 item 4).

---

## Patch 3: no demo archetype exercises the limited-history path

All six archetypes in `03` §2 carry 273+ days of history by design. Nothing in the default demo data ever triggers the fallback banner or `limited_history: true`, which means that path could ship un-exercised. **Fixed**: added a note directly under the archetype table in `03` §2 flagging this as a real path that needs a manual test (e.g. a truncated slice of one archetype's own history), not a gap to leave to whichever SKU happens to be shortest.

---

## How to fold this into your Implementation Plan

Your plan document doesn't need to be regenerated, five targeted edits close the gaps this patch and the earlier review surfaced:

1. **Verification step 2 — stop checking literal digits.** Replace:
   > "Walk SKU-101 through Screen 2 → PO → Screen 3, verify IP=84+144=228, ROP=86, status=Healthy"

   with a relational check that holds regardless of the exact numbers a live-generated dataset produces:
   > "Walk SKU-101 through Screen 2 → PO → Screen 3. Confirm `inventory_position == current_stock + on_order_stock` at each step, confirm `status`/`new_status` matches `classify_status(ip, rop)` under the IP/ROP actually computed, and confirm `safety_stock == ceil(Z_beverage × daily_sales_std × √lead_time_days)` using the SKU's own live `daily_sales_std`. Don't hardcode 40/11/49/144, those are illustrative reference values (Patch v3.1), not a literal target for the generated dataset."

2. **Section 7 (`forecast_router.py`) — place `limited_history` explicitly.** Add: "Set `limited_history` (top-level, per `02` §3.1 Patch v3.1) whenever `len(history) < 14` or every model score is `None`; the frontend reads this flag directly rather than inferring it from array lengths."

3. **Section 7, step 8 — name the `daily_forecast` argument.** Add after step 8: "`daily_forecast` passed to `InventoryRecommender.calculate()` is `quantiles["P50"]` from step 6, not raw history."

4. **Section 8 — drop the caching hedge.** Change the `[!NOTE]` callout's "re-run the forecast pipeline for this SKU (or cache it)" to just "re-run the forecast pipeline for this SKU." The request payload only carries `sku_id` and `quantity`, there's nothing to key a cache on, and `03` §4 already confirms recompute is fine at demo scale.

5. **Non-Negotiable Constraint Checklist — add the leakage rule.** It's the most heavily emphasized rule in `03` (§1) and the entire reason `_model_point_forecast` is designed the way it is, but it's the one major rule missing from your table. Add a row: `No model backtest function may see the actual values of the window it's scored against | ✅ run_backtest calls _model_point_forecast identically to live forecasting (03 §1, 02 §2.2)`.

6. **Section 6 (generator table) — note the short-history gap.** Add a line after the archetype table: "None of the six archetypes has short history by design (Patch v3.1, `03` §2); add a manual test with a truncated history slice to exercise the `limited_history` fallback before calling verification complete."

None of these six edits touch your directory structure, your service-by-service breakdown, or your non-negotiable constraints table's existing rows, they're additive.
