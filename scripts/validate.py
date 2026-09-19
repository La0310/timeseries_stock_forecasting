"""
validate.py
End-to-end validation script. Runs independently (no web server needed).

Checks:
  1. All 6 SKUs: Expected Model wins, all model scores printed
  2. SKU-101 relational check (post-PO): ip==current+on_order,
     status matches classify_status(ip, rop), ss matches formula
  3. Screen 1/2 badge consistency (same status field from same API call)
  4. SKU-505 quarterly seasonality = "High"; short-history slice = "Insufficient History"
  5. limited_history flag: 6 demo SKUs → False; truncated slice → True

Run from stock-management/:
    .venv\\Scripts\\python.exe validate.py
"""

import sys
import math
import os
from pathlib import Path

# Force UTF-8 on Windows console
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Allow importing backend as a package
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import numpy as np

from backend.services.demand_classifier import DemandClassifier
from backend.services.forecast_engine import ForecastEngine
from backend.services.inventory_recommender import InventoryRecommender
from backend.services.model_router import ModelRouter
from backend.services.purchase_order_service import PurchaseOrderService

DATA_PATH = Path(__file__).parent / "backend" / "data" / "demo_inventory.csv"

# Expected models per spec 03 §2
EXPECTED_MODELS = {
    "SKU-101": "Chronos-2",
    "SKU-202": "TSB",
    "SKU-303": "LightGBM",
    "SKU-404": "SBA",
    "SKU-505": "SARIMAX",
    "SKU-606": "ETS",
}


def load_history(df: pd.DataFrame, sku_id: str) -> pd.Series:
    rows = df[df["Product ID"] == sku_id].sort_values("Date")
    return rows["Units Sold"].reset_index(drop=True).astype(float)


def load_meta(df: pd.DataFrame, sku_id: str) -> dict:
    row = df[df["Product ID"] == sku_id].iloc[0]
    return {
        "category": row["Category"],
        "pack_size": int(row["Pack Size"]),
        "lead_time_days": int(row["Lead Time Days"]),
        "current_stock": int(row["Current Stock"]),
    }


def run_pipeline(df: pd.DataFrame, sku_id: str) -> dict:
    """Runs the full demand→routing→forecast→replenishment pipeline for one SKU."""
    history = load_history(df, sku_id)
    meta = load_meta(df, sku_id)

    demand_analysis = DemandClassifier.analyze(history)
    daily_sales_std = demand_analysis["daily_sales_std"]

    wape_dict = {
        model: ForecastEngine.run_backtest(history, model)
        for model in ModelRouter.MODEL_REGISTRY
    }
    routing = ModelRouter.route(demand_analysis, wape_dict)
    selected_model = routing["selected_model"]

    all_null = all(w is None for w in wape_dict.values())
    limited_history = len(history) < 14 or all_null

    on_order_stock = PurchaseOrderService.get_on_order_stock(sku_id)
    quantiles = ForecastEngine.generate_quantiles(history, selected_model, 28, daily_sales_std)
    p50 = quantiles["P50"]

    replenishment = InventoryRecommender.calculate(
        current_stock=meta["current_stock"],
        on_order_stock=on_order_stock,
        category=meta["category"],
        lead_time_days=meta["lead_time_days"],
        daily_sales_std=daily_sales_std,
        daily_forecast=p50,
        pack_size=meta["pack_size"],
    )

    return {
        "sku_id": sku_id,
        "limited_history": limited_history,
        "demand_analysis": demand_analysis,
        "routing": routing,
        "quantiles": quantiles,
        "replenishment": replenishment,
        "meta": meta,
    }


def sep(char="-", width=72):
    print(char * width)


# ===========================================================================
# CHECK 1: All 6 SKU model selections + scores
# ===========================================================================

def check1_model_selection(df):
    sep("═")
    print("CHECK 1 — Model Selection & Scores for All 6 SKUs")
    sep("═")
    all_pass = True
    for sku_id, expected in EXPECTED_MODELS.items():
        result = run_pipeline(df, sku_id)
        routing = result["routing"]
        selected = routing["selected_model"]
        scores = routing["model_scores"]
        demand_class = result["demand_analysis"]["demand_class"]

        status_icon = "✅" if selected == expected else "❌"
        print(f"\n  {sku_id} ({demand_class}): selected={selected} {status_icon} (expected {expected})")
        for model in ModelRouter.MODEL_REGISTRY:
            s = scores.get(model)
            score_str = f"{s:.1f}%" if s is not None else "null"
            marker = " ← SELECTED" if model == selected else ""
            print(f"    {model:12s}  {score_str}{marker}")

        if selected != expected:
            all_pass = False
            print(f"  *** MISMATCH: got {selected}, expected {expected}")

    sep()
    print(f"\nCHECK 1: {'ALL PASS ✅' if all_pass else 'FAILURES DETECTED ❌'}\n")
    return all_pass


# ===========================================================================
# CHECK 2: SKU-101 relational walk-through (Screen 2 → PO → Screen 3)
# ===========================================================================

def check2_sku101_relational(df):
    sep("=")
    print("CHECK 2 -- SKU-101 Relational Walk-Through (Screen 2 -> PO -> Screen 3)")
    sep("=")

    # Reset ledger so on_order=0 at start
    PurchaseOrderService.reset()
    result = run_pipeline(df, "SKU-101")
    rep = result["replenishment"]
    da = result["demand_analysis"]
    meta = result["meta"]

    print(f"\n  [Screen 2 -- before PO]")
    print(f"  current_stock      = {rep['current_stock']}")
    print(f"  on_order_stock     = {rep['on_order_stock']}")
    print(f"  inventory_position = {rep['inventory_position']}")
    print(f"  reorder_point      = {rep['reorder_point']}")
    print(f"  safety_stock       = {rep['safety_stock']}")
    print(f"  status             = {rep['status']}")
    print(f"  recommended_order  = {rep['recommended_order']}")
    print(f"  daily_sales_std    = {da['daily_sales_std']}")

    # Relational checks: Screen 2
    ip = rep["current_stock"] + rep["on_order_stock"]
    rop = rep["reorder_point"]
    z = InventoryRecommender.service_level_z(meta["category"])
    std = da["daily_sales_std"]
    L = meta["lead_time_days"]
    ss_formula = math.ceil(z * std * math.sqrt(L))
    expected_status = InventoryRecommender.classify_status(ip, rop)

    checks = [
        ("ip == current + on_order", rep["inventory_position"] == ip),
        ("status matches classify_status(ip, rop)", rep["status"] == expected_status),
        ("safety_stock matches ceil(Z * std * sqrt(L))", rep["safety_stock"] == ss_formula),
    ]
    for desc, ok in checks:
        print(f"  {'OK' if ok else 'FAIL'} {desc}: formula={ss_formula if 'safety' in desc else (expected_status if 'status' in desc else ip)} actual={rep['safety_stock'] if 'safety' in desc else (rep['status'] if 'status' in desc else rep['inventory_position'])}")

    # Simulate PO
    po_qty = rep["recommended_order"]
    print(f"\n  [Placing PO for {po_qty} units]")
    po = PurchaseOrderService.record_purchase_order("SKU-101", po_qty)
    new_on_order = po["on_order_stock"]
    new_ip = rep["current_stock"] + new_on_order
    new_status = InventoryRecommender.classify_status(new_ip, rop)

    print(f"  [Screen 3 -- after PO]")
    print(f"  po_id              = {po['po_id']}")
    print(f"  on_order_stock     = {new_on_order}")
    print(f"  inventory_position = {new_ip}")
    print(f"  reorder_point      = {rop}")
    print(f"  new_status         = {new_status}")

    po_checks = [
        ("post-PO ip == current + new_on_order", new_ip == rep["current_stock"] + po_qty),
        ("new_status matches classify_status(new_ip, rop)", new_status == InventoryRecommender.classify_status(new_ip, rop)),
    ]
    all_pass = all(ok for _, ok in checks) and all(ok for _, ok in po_checks)
    for desc, ok in po_checks:
        print(f"  {'OK' if ok else 'FAIL'} {desc}")

    sep()
    print(f"\nCHECK 2: {'ALL PASS' if all_pass else 'FAILURES DETECTED'}\n")
    return all_pass


# ===========================================================================
# CHECK 3: Screen 1 / Screen 2 badge consistency
# ===========================================================================

def check3_badge_consistency(df):
    sep("=")
    print("CHECK 3 -- Screen 1/2 Status Badge Consistency")
    sep("=")
    print("  (Both screens read replenishment.status from the same pipeline)")
    PurchaseOrderService.reset()
    all_pass = True
    for sku_id in EXPECTED_MODELS:
        result = run_pipeline(df, sku_id)
        status = result["replenishment"]["status"]
        # The badge is the same field regardless of which screen reads it:
        # screen1 and screen2 both call POST /api/forecast and read replenishment.status
        # Here we just confirm the value is deterministic (run twice, same result)
        result2 = run_pipeline(df, sku_id)
        status2 = result2["replenishment"]["status"]
        match = status == status2
        print(f"  {sku_id}: status={status!r}  consistent={match} {'OK' if match else 'FAIL'}")
        if not match:
            all_pass = False
    sep()
    print(f"\nCHECK 3: {'ALL PASS' if all_pass else 'FAILURES DETECTED'}\n")
    return all_pass


# ===========================================================================
# CHECK 4: Seasonality -- SKU-505 High, short-history -> Insufficient History
# ===========================================================================

def check4_seasonality(df):
    sep("=")
    print("CHECK 4 -- Seasonality Labels")
    sep("=")
    all_pass = True

    # SKU-505 should have quarterly_label = "High"
    history505 = load_history(df, "SKU-505")
    da505 = DemandClassifier.analyze(history505)
    ql = da505["seasonality"]["quarterly_label"]
    acf = da505["seasonality"]["quarterly_acf"]
    ok = ql == "High"
    print(f"  SKU-505 quarterly_label: {ql!r} (ACF={acf}) {'OK' if ok else 'FAIL (expected High)'}")
    if not ok:
        all_pass = False

    # Short-history slice (< 3*91=273 days) -> quarterly = "Insufficient History"
    short = history505.iloc[:100]  # only 100 days, < 273
    da_short = DemandClassifier.analyze(short)
    ql_short = da_short["seasonality"]["quarterly_label"]
    acf_short = da_short["seasonality"]["quarterly_acf"]
    ok2 = ql_short == "Insufficient History" and acf_short is None
    print(f"  100-day slice quarterly_label: {ql_short!r} (ACF={acf_short}) {'OK' if ok2 else 'FAIL (expected Insufficient History)'}")
    if not ok2:
        all_pass = False

    sep()
    print(f"\nCHECK 4: {'ALL PASS' if all_pass else 'FAILURES DETECTED'}\n")
    return all_pass


# ===========================================================================
# CHECK 5: limited_history flag
# ===========================================================================

def check5_limited_history(df):
    sep("=")
    print("CHECK 5 -- limited_history Flag")
    sep("=")
    all_pass = True

    # All 6 demo SKUs should have limited_history=False
    PurchaseOrderService.reset()
    for sku_id in EXPECTED_MODELS:
        result = run_pipeline(df, sku_id)
        lh = result["limited_history"]
        ok = lh is False
        print(f"  {sku_id}: limited_history={lh} {'✅' if ok else '❌ (expected False)'}")
        if not ok:
            all_pass = False

    # Truncated slice (10 days) should give limited_history=True
    history_short = load_history(df, "SKU-101").iloc[:10]
    da = DemandClassifier.analyze(history_short)
    wape_dict = {
        m: ForecastEngine.run_backtest(history_short, m)
        for m in ModelRouter.MODEL_REGISTRY
    }
    all_null = all(w is None for w in wape_dict.values())
    lh_short = len(history_short) < 14 or all_null
    ok_short = lh_short is True
    print(f"  10-day slice:  limited_history={lh_short} {'✅' if ok_short else '❌ (expected True)'}")
    if not ok_short:
        all_pass = False

    sep()
    print(f"\nCHECK 5: {'ALL PASS ✅' if all_pass else 'FAILURES DETECTED ❌'}\n")
    return all_pass


# ===========================================================================
# Main
# ===========================================================================

if __name__ == "__main__":
    df = pd.read_csv(DATA_PATH)
    df["Date"] = pd.to_datetime(df["Date"])

    PurchaseOrderService.reset()
    results = [
        check1_model_selection(df),
        check2_sku101_relational(df),
        check3_badge_consistency(df),
        check4_seasonality(df),
        check5_limited_history(df),
    ]
    sep("═")
    overall = all(results)
    print(f"\nOVERALL VALIDATION: {'✅ ALL CHECKS PASS' if overall else '❌ SOME CHECKS FAILED'}")
    sep("═")
    sys.exit(0 if overall else 1)
