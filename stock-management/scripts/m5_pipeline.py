"""
m5_pipeline.py
M5 Demand Dataset → demo_inventory.csv enrichment pipeline.

What this does
--------------
1. Tries to load M5 from Kaggle (if credentials + kaggle CLI available).
   Falls back to a built-in synthetic M5-like generator if not.
2. Selects up to MAX_SKUS representative item-store time series.
3. Aggregates to daily demand per pseudo-SKU (already daily in M5).
4. Enriches non-demand fields with Faker following business constraints.
5. Writes backend/data/demo_inventory.csv (same schema as before).

Usage
-----
    # Run from the stock-management/ project root:
    python scripts/m5_pipeline.py          # generate with Faker fallback
    python scripts/m5_pipeline.py --kaggle # require real M5 data

Business-constraint rules for Faker fields
------------------------------------------
Category  | Lead time (days) | Pack size | Unit price range
Beverage  | 2–5              | 12 or 24  | $1–$8
Office    | 3–7              | 1, 6, 12  | $2–$50
Hardware  | 7–14             | 1 or 6    | $10–$200
MRO/Parts | 14–30            | 1         | $5–$500
Seasonal  | 5–10             | 6 or 12   | $3–$80
"""

from __future__ import annotations

import argparse
import hashlib
import random
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from faker import Faker
    _HAS_FAKER = True
except ImportError:
    _HAS_FAKER = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DATA_DIR = Path(__file__).parent
OUTPUT_CSV = DATA_DIR / "demo_inventory.csv"

MAX_SKUS = 30          # SKUs in output (keeps file manageable)
HISTORY_DAYS = 365     # one year of daily history per SKU
RANDOM_SEED = 42

CATEGORIES = ["Beverage", "Office", "Hardware", "MRO / Parts", "Seasonal overlay"]

# Business constraints per category
CATEGORY_CONSTRAINTS: dict[str, dict] = {
    "Beverage":         {"lead": (2, 5),   "pack": [12, 24],    "price": (1.0,   8.0)},
    "Office":           {"lead": (3, 7),   "pack": [1, 6, 12],  "price": (2.0,  50.0)},
    "Hardware":         {"lead": (7, 14),  "pack": [1, 6],      "price": (10.0, 200.0)},
    "MRO / Parts":      {"lead": (14, 30), "pack": [1],         "price": (5.0,  500.0)},
    "Seasonal overlay": {"lead": (5, 10),  "pack": [6, 12],     "price": (3.0,   80.0)},
}

# M5 category → our category mapping
M5_CAT_MAP = {
    "FOODS":     "Beverage",
    "HOUSEHOLD": "Office",
    "HOBBIES":   "Hardware",
}


# ---------------------------------------------------------------------------
# Synthetic M5-like generator (fallback when Kaggle unavailable)
# ---------------------------------------------------------------------------

def _demand_profile(rng: np.random.Generator, category: str, n_days: int) -> np.ndarray:
    """
    Generate a realistic daily demand series for the given category.
    Each category has a distinct ADI/CV² profile matching M5 archetypes.
    """
    base_rates = {
        "Beverage":         12.0,   # high freq, low CV² → Smooth
        "Office":           4.0,    # medium freq → Intermittent
        "Hardware":         8.0,    # high freq, high CV² → Erratic
        "MRO / Parts":      1.5,    # low freq, high CV² → Lumpy
        "Seasonal overlay": 6.0,    # seasonal sinusoidal
    }
    cv2_shapes = {
        "Beverage":         0.2,
        "Office":           0.4,
        "Hardware":         1.2,
        "MRO / Parts":      2.5,
        "Seasonal overlay": 0.8,
    }
    rate = base_rates.get(category, 5.0)
    cv2  = cv2_shapes.get(category, 0.5)

    # Seasonal overlay: add sinusoidal component
    if category == "Seasonal overlay":
        seasonal = 0.5 * np.sin(2 * np.pi * np.arange(n_days) / 91)
        rate_arr = np.maximum(0.5, rate * (1 + seasonal))
    else:
        rate_arr = np.full(n_days, rate)

    # Negative binomial: mean = rate, variance = mean + mean²/r → cv² ≈ 1/r
    r = max(0.5, 1.0 / cv2) if cv2 > 0 else 10.0
    p = r / (r + rate_arr)

    # Intermittent: randomly zero out days (MRO/Office)
    demand = rng.negative_binomial(r, p).astype(float)
    if category in ("MRO / Parts", "Office"):
        zero_mask = rng.random(n_days) < (0.6 if category == "MRO / Parts" else 0.35)
        demand[zero_mask] = 0.0

    return demand


def _generate_synthetic_m5(n_skus: int = MAX_SKUS) -> list[dict]:
    """
    Returns a list of dicts, one per SKU, with keys:
        sku_id, category, demand (np.ndarray of length HISTORY_DAYS)
    """
    rng = np.random.default_rng(RANDOM_SEED)
    records = []
    cats_cycle = CATEGORIES * (n_skus // len(CATEGORIES) + 1)
    for i in range(n_skus):
        cat = cats_cycle[i]
        records.append({
            "sku_id": f"SKU-{(i + 1) * 101:04d}",
            "category": cat,
            "demand": _demand_profile(rng, cat, HISTORY_DAYS),
        })
    return records


# ---------------------------------------------------------------------------
# Real M5 loader (Kaggle path)
# ---------------------------------------------------------------------------

def _load_m5_from_kaggle(kaggle_dir: Path) -> list[dict]:
    """
    Load M5 data from a pre-downloaded Kaggle directory.
    Expects: sales_train_validation.csv, calendar.csv in kaggle_dir.
    """
    sales_path = kaggle_dir / "sales_train_validation.csv"
    cal_path   = kaggle_dir / "calendar.csv"

    if not sales_path.exists():
        raise FileNotFoundError(f"M5 sales file not found: {sales_path}")

    print(f"[m5_pipeline] Loading M5 from {kaggle_dir} ...")
    sales = pd.read_csv(sales_path)
    cal   = pd.read_csv(cal_path)

    # Narrow to day columns only
    day_cols = [c for c in sales.columns if c.startswith("d_")]
    # Keep only up to HISTORY_DAYS most recent days
    day_cols = day_cols[-HISTORY_DAYS:]

    rng = random.Random(RANDOM_SEED)
    records = []
    # Sample evenly across M5 categories
    sampled = sales.groupby("cat_id").apply(
        lambda g: g.sample(min(MAX_SKUS // 3, len(g)), random_state=RANDOM_SEED)
    ).reset_index(drop=True).head(MAX_SKUS)

    for _, row in sampled.iterrows():
        cat = M5_CAT_MAP.get(row.get("cat_id", "FOODS"), "Beverage")
        item_id = row.get("item_id", "UNKNOWN")
        store_id = row.get("store_id", "TX_1")
        sku_id = f"M5-{item_id[:8]}-{store_id}"[:16]
        demand = row[day_cols].values.astype(float)
        records.append({"sku_id": sku_id, "category": cat, "demand": demand})

    return records


# ---------------------------------------------------------------------------
# Faker enrichment
# ---------------------------------------------------------------------------

def _enrich_with_faker(records: list[dict], seed: int = RANDOM_SEED) -> pd.DataFrame:
    """
    Takes list of {sku_id, category, demand} dicts.
    Expands each into HISTORY_DAYS rows with all inventory CSV columns.
    Business-constraint fields are generated with Faker + seeded RNG.
    """
    if not _HAS_FAKER:
        raise ImportError("faker is required: uv pip install faker>=25.0")

    fake = Faker()
    Faker.seed(seed)
    rng = random.Random(seed)

    start_date = pd.Timestamp("2025-01-01")
    date_range = pd.date_range(start_date, periods=HISTORY_DAYS, freq="D")

    rows: list[dict] = []
    for rec in records:
        sku_id   = rec["sku_id"]
        category = rec["category"]
        demand   = rec["demand"]
        c        = CATEGORY_CONSTRAINTS[category]

        # Per-SKU stable fields (same for all 365 rows)
        product_name  = fake.catch_phrase()[:40]
        supplier      = fake.company()[:30]
        lead_time     = rng.randint(*c["lead"])
        pack_size     = rng.choice(c["pack"])
        unit_price    = round(rng.uniform(*c["price"]), 2)
        discount_pct  = round(rng.uniform(0.0, 0.25), 2)
        reorder_qty   = pack_size * rng.randint(2, 8)
        safety_buffer = rng.randint(5, 30)

        # Current stock: simulate as remaining after last 30d demand
        avg_daily = float(np.mean(demand[-30:])) if len(demand) >= 30 else float(np.mean(demand))
        current_stock_base = int(avg_daily * lead_time * 1.5 + safety_buffer)

        for i, (date, units_sold) in enumerate(zip(date_range, demand)):
            # Current stock decreases linearly through the year (simplified)
            decay = max(0, current_stock_base - int(avg_daily * (i / HISTORY_DAYS) * 30))
            rows.append({
                "Date":             date,
                "Product ID":       sku_id,
                "Product Name":     product_name,
                "Category":         category,
                "Units Sold":       int(units_sold),
                "Current Stock":    decay,
                "Pack Size":        pack_size,
                "Lead Time Days":   lead_time,
                "Unit Price":       unit_price,
                "Discount":         discount_pct,
                "Supplier":         supplier,
                "Reorder Qty":      reorder_qty,
                "Safety Stock":     safety_buffer,
                # Constant placeholders (zero historical variance — see scenario_engine.py)
                "Price":            unit_price,
                "Competitor Pricing": round(unit_price * rng.uniform(0.85, 1.15), 2),
                "Holiday/Promotion": 0,
                "Weather Condition": "Normal",
            })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def build_m5_dataset(use_kaggle: bool = False, kaggle_dir: Path | None = None) -> pd.DataFrame:
    """
    Build the enriched M5-backed demo inventory DataFrame.

    Parameters
    ----------
    use_kaggle : if True, fail loudly if Kaggle data is unavailable.
    kaggle_dir : path to pre-downloaded M5 Kaggle files.
    """
    kaggle_dir = kaggle_dir or DATA_DIR / "m5_kaggle"

    if use_kaggle:
        records = _load_m5_from_kaggle(kaggle_dir)
        print(f"[m5_pipeline] Loaded {len(records)} SKUs from Kaggle M5.")
    else:
        if kaggle_dir.exists() and (kaggle_dir / "sales_train_validation.csv").exists():
            try:
                records = _load_m5_from_kaggle(kaggle_dir)
                print(f"[m5_pipeline] Loaded {len(records)} SKUs from cached Kaggle M5.")
            except Exception as e:
                print(f"[m5_pipeline] Kaggle load failed ({e}), using synthetic M5-like data.")
                records = _generate_synthetic_m5()
        else:
            print("[m5_pipeline] No Kaggle M5 found — generating synthetic M5-like time series.")
            records = _generate_synthetic_m5()

    print(f"[m5_pipeline] Enriching {len(records)} SKUs with Faker business fields ...")
    df = _enrich_with_faker(records)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"[m5_pipeline] Wrote {len(df):,} rows to {OUTPUT_CSV}")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="M5 inventory dataset generator")
    parser.add_argument("--kaggle", action="store_true", help="Require real Kaggle M5 data")
    parser.add_argument(
        "--kaggle-dir", type=Path, default=None,
        help="Path to Kaggle M5 CSV directory (default: backend/data/m5_kaggle/)"
    )
    args = parser.parse_args()
    build_m5_dataset(use_kaggle=args.kaggle, kaggle_dir=args.kaggle_dir)
