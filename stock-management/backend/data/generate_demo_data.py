"""
generate_demo_data.py
Synthetic daily-sales generator for the 6 demo SKU archetypes.
Spec: 03_addendum_spec_v3.md §2

Run once to regenerate demo_inventory.csv:
    python backend/data/generate_demo_data.py

Design rules:
  - Data is generated so that DemandClassifier.analyze() NATURALLY produces
    the target ADI/CV² quadrant for each SKU. No hardcoding of classifier
    output. The numbers fall out of the data.
  - SKU-505 must produce lag-91 ACF >= 0.6 from its own data.
  - All SKUs carry >= 365 days of history (>= 3×91=273 required for quarterly ACF).
  - Schema includes all columns referenced by 03 §1 (leakage rule applies:
    Demand Forecast and Units Ordered are present for schema completeness only
    and are never read by any service module).
"""

import numpy as np
import pandas as pd
from pathlib import Path

RNG = np.random.default_rng(seed=42)

# ---------------------------------------------------------------------------
# SKU metadata
# ---------------------------------------------------------------------------
SKUS = {
    "SKU-101": {
        "product_name": "Coca-Cola 330ml Can",
        "category": "Beverage",
        "pack_size": 24,
        "lead_time_days": 7,
        "store_id": "STORE-01",
        "region": "North",
        "price": 1.20,
        "initial_stock": 40,
        # Target: ADI < 1.32, CV² < 0.49 → Smooth
        # Strategy: near-daily Poisson, low variance
        "generator": "smooth_poisson",
        "gen_params": {"lam": 5.4, "zero_prob": 0.02, "n_days": 365},
    },
    "SKU-202": {
        "product_name": "HP LaserJet Drum Unit",
        "category": "Office",
        "pack_size": 1,
        "lead_time_days": 14,
        "store_id": "STORE-01",
        "region": "North",
        "price": 89.99,
        "initial_stock": 8,
        # Target: ADI >= 1.32, CV² < 0.49 → Intermittent
        # Strategy: Bernoulli occurrence (p≈0.36) × stable Poisson size
        "generator": "intermittent",
        "gen_params": {"occur_prob": 0.32, "size_lam": 5, "n_days": 365},
    },
    "SKU-303": {
        "product_name": "DeWalt 18V Cordless Drill",
        "category": "Hardware",
        "pack_size": 4,
        "lead_time_days": 21,
        "store_id": "STORE-01",
        "region": "North",
        "price": 149.00,
        "initial_stock": 28,
        # Target: ADI < 1.32, CV² >= 0.49 → Erratic
        # Strategy: near-daily (p≈0.91), mixture of baseline + promo spikes
        "generator": "erratic",
        "gen_params": {
            "occur_prob": 0.91,
            "base_lam": 3,
            "spike_prob": 0.08,
            "spike_low": 20,
            "spike_high": 45,
            "n_days": 365,
        },
    },
    "SKU-404": {
        "product_name": "Industrial 4-Inch Gate Valve",
        "category": "MRO / Parts",
        "pack_size": 1,
        "lead_time_days": 30,
        "store_id": "STORE-01",
        "region": "North",
        "price": 320.00,
        "initial_stock": 5,
        # Target: ADI >= 1.32, CV² >= 0.49 → Lumpy
        # Strategy: sparse Bernoulli (p≈0.29) × high-variance Gamma size
        "generator": "lumpy",
        "gen_params": {
            "occur_prob": 0.294,
            "size_shape": 0.5,
            "size_scale": 6.0,
            "n_days": 365,
        },
    },
    "SKU-505": {
        "product_name": "LED Fairy String Lights",
        "category": "Seasonal overlay",
        "pack_size": 12,
        "lead_time_days": 14,
        "store_id": "STORE-01",
        "region": "North",
        "price": 12.99,
        "initial_stock": 60,
        # Target: ADI ≈ 1.05, CV² ≈ 0.24, lag-91 ACF >= 0.6 → Smooth + High seasonal
        # Strategy: deterministic sin-wave seasonal component (no Poisson noise)
        # layered on a stable base. High amplitude-to-noise ratio is needed
        # so lag-91 ACF stays >= 0.6. Reduced zero_prob keeps ADI low.
        # History >= 3×91=273 days (we use 366) so ACF is estimated, not null.
        "generator": "seasonal",
        "gen_params": {
            "base_lam": 14.0,
            "seasonal_amplitude": 10.0,
            "seasonal_period": 91,
            "zero_prob": 0.05,
            "n_days": 730,
        },
    },
    "SKU-606": {
        "product_name": "Multipurpose Copy Paper A4",
        "category": "Office",
        "pack_size": 5,
        "lead_time_days": 7,
        "store_id": "STORE-01",
        "region": "North",
        "price": 6.50,
        "initial_stock": 150,
        # Target: ADI ≈ 1.00, CV² ≈ 0.08 → High-Volume Smooth
        # Strategy: very stable daily sales (near-zero variance)
        "generator": "high_vol_smooth",
        "gen_params": {"mean": 21, "std": 1.5, "n_days": 365},
    },
}

START_DATE = pd.Timestamp("2025-09-07")


# ---------------------------------------------------------------------------
# Per-archetype generators
# ---------------------------------------------------------------------------


def gen_smooth_poisson(params: dict) -> np.ndarray:
    """Near-daily Poisson sales. zero_prob fraction of days are forced zero
    to produce ADI slightly above 1.0 and keep CV² low.
    Target: ADI ≈ 1.02, CV² ≈ 0.18"""
    n = params["n_days"]
    lam = params["lam"]
    zero_prob = params.get("zero_prob", 0.02)
    sales = RNG.poisson(lam=lam, size=n).astype(float)
    zero_mask = RNG.random(n) < zero_prob
    sales[zero_mask] = 0.0
    return sales


def gen_intermittent(params: dict) -> np.ndarray:
    """Bernoulli occurrence × Poisson size.
    Target: ADI ≈ 2.80 (occur_prob≈0.357), CV² ≈ 0.22"""
    n = params["n_days"]
    occur = RNG.random(n) < params["occur_prob"]
    sizes = RNG.poisson(lam=params["size_lam"], size=n)
    # Add 1 to avoid zero sizes on demand days (keeps CV² low)
    sizes = np.maximum(sizes, 1)
    return (occur * sizes).astype(float)


def gen_erratic(params: dict) -> np.ndarray:
    """Near-daily occurrence, mixture: baseline Poisson + large promo spikes.
    Target: ADI ≈ 1.10, CV² ≈ 0.85"""
    n = params["n_days"]
    occur = RNG.random(n) < params["occur_prob"]
    base = RNG.poisson(lam=params["base_lam"], size=n)
    base = np.maximum(base, 1)  # at least 1 on demand days
    spikes = RNG.random(n) < params["spike_prob"]
    spike_sizes = RNG.integers(
        params["spike_low"], params["spike_high"] + 1, size=n
    ).astype(float)
    sales = occur * (base + spikes * spike_sizes)
    return sales.astype(float)


def gen_lumpy(params: dict) -> np.ndarray:
    """Sparse occurrence × high-variance Gamma size.
    Target: ADI ≈ 3.40, CV² ≈ 1.12"""
    n = params["n_days"]
    occur = RNG.random(n) < params["occur_prob"]
    # Gamma with shape < 1 gives heavy right tail → high CV²
    sizes = RNG.gamma(
        shape=params["size_shape"], scale=params["size_scale"], size=n
    )
    sizes = np.round(sizes).astype(float)
    sizes = np.maximum(sizes, 1)  # at least 1 on demand days
    return (occur * sizes).astype(float)


def gen_seasonal(params: dict) -> np.ndarray:
    """Deterministic sin-wave seasonal base + tiny Gaussian noise.

    Key design: the seasonal component is DETERMINISTIC (no Poisson noise),
    so the autocorrelation signal at lag-91 is strong (ACF >= 0.6).
    Only a tiny Gaussian noise term (std=0.8) is added to simulate
    day-to-day measurement variation.

    Keeping the noise small relative to the seasonal swing means:
    - Non-zero conditional values all follow the cycle closely → CV² < 0.49
    - The lag-91 autocorrelation is dominated by the sin component → ACF >= 0.6
    - ADI stays near 1.05 because floor keeps most days non-zero

    Target: ADI ≈ 1.05, CV² < 0.49, lag-91 ACF >= 0.6
    """
    n = params["n_days"]
    base = params["base_lam"]
    amp = params["seasonal_amplitude"]
    period = params["seasonal_period"]
    zero_prob = params.get("zero_prob", 0.04)

    t = np.arange(n)
    # Purely deterministic seasonal signal (full sine cycle every 91 days)
    seasonal_signal = amp * np.sin(2 * np.pi * t / period)

    # Deterministic signal + tiny Gaussian noise to avoid a perfectly flat series
    noise = RNG.normal(loc=0.0, scale=0.8, size=n)
    raw = base + seasonal_signal + noise

    # Round to integers; floor at 1 so most days have a sale (low ADI)
    sales = np.maximum(1.0, np.round(raw)).astype(float)

    # Small fraction of zero-demand days to target ADI ≈ 1.05
    zero_mask = RNG.random(n) < zero_prob
    sales[zero_mask] = 0.0
    return sales


def gen_high_vol_smooth(params: dict) -> np.ndarray:
    """Very stable daily sales — near-zero variance.
    Target: ADI ≈ 1.00, CV² ≈ 0.08"""
    n = params["n_days"]
    # Truncated normal to avoid zeros (all days sell)
    raw = RNG.normal(loc=params["mean"], scale=params["std"], size=n)
    return np.maximum(1.0, np.round(raw)).astype(float)


GENERATORS = {
    "smooth_poisson": gen_smooth_poisson,
    "intermittent": gen_intermittent,
    "erratic": gen_erratic,
    "lumpy": gen_lumpy,
    "seasonal": gen_seasonal,
    "high_vol_smooth": gen_high_vol_smooth,
}


# ---------------------------------------------------------------------------
# Verification helper (print to stdout at generation time)
# ---------------------------------------------------------------------------


def verify_sku(sku_id: str, sales: np.ndarray) -> None:
    """Compute and print ADI, CV², and lag-91 ACF so generator accuracy
    can be eyeballed without running the full app."""
    n = len(sales)
    non_zero = sales[sales > 0]
    adi = n / len(non_zero) if len(non_zero) > 0 else float("inf")
    cv2 = (
        (np.std(non_zero, ddof=1) / np.mean(non_zero)) ** 2
        if len(non_zero) > 1
        else 0.0
    )
    daily_std = np.std(sales, ddof=1)

    # Lag-91 ACF (same formula as demand_classifier.py)
    acf91 = None
    if n >= 3 * 91:
        y = sales - sales.mean()
        denom = np.sum(y ** 2)
        acf91 = float(np.sum(y[: n - 91] * y[91:]) / denom) if denom != 0 else 0.0

    print(
        f"{sku_id}: n={n}, ADI={adi:.2f}, CV²={cv2:.2f}, "
        f"daily_std={daily_std:.2f}, lag91_ACF={acf91}"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def build_csv() -> pd.DataFrame:
    all_rows = []

    for sku_id, meta in SKUS.items():
        gen_fn = GENERATORS[meta["generator"]]
        sales = gen_fn(meta["gen_params"])
        n = len(sales)

        dates = pd.date_range(START_DATE - pd.Timedelta(days=n - 1), periods=n, freq="D")

        # Compute a rolling inventory level (for schema completeness only)
        stock = float(meta["initial_stock"])
        inv_levels = []
        for units in sales:
            stock = max(0.0, stock - units)
            inv_levels.append(int(stock))

        rows = pd.DataFrame(
            {
                "Date": dates.strftime("%Y-%m-%d"),
                "Store ID": meta["store_id"],
                "Product ID": sku_id,
                "Product Name": meta["product_name"],
                "Category": meta["category"],
                "Region": meta["region"],
                "Inventory Level": inv_levels,
                "Units Sold": sales.astype(int),
                # Schema-completeness columns — NEVER used as model features (03 §1)
                "Units Ordered": 0,
                "Demand Forecast": np.round(sales * 0.95, 1),  # synthetic, schema-only
                "Price": meta["price"],
                "Discount": 0.0,
                "Weather Condition": "Normal",
                "Holiday/Promotion": 0,
                "Competitor Pricing": round(meta["price"] * 1.05, 2),
                "Seasonality": (
                    "Quarterly" if meta["generator"] == "seasonal" else "None"
                ),
                # Extra metadata columns used by the API to avoid a separate lookup
                "Pack Size": meta["pack_size"],
                "Lead Time Days": meta["lead_time_days"],
                "Current Stock": meta["initial_stock"],
            }
        )
        all_rows.append(rows)
        verify_sku(sku_id, sales)

    df = pd.concat(all_rows, ignore_index=True)
    return df


if __name__ == "__main__":
    out_path = Path(__file__).parent / "demo_inventory.csv"
    df = build_csv()
    df.to_csv(out_path, index=False)
    print(f"\nWrote {len(df):,} rows -> {out_path}")
