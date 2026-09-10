"""
Diagnostic v3: fine-grained search for SARIMAX and TSB winners.
"""
import sys
sys.path.insert(0, ".")

import numpy as np
import pandas as pd
from backend.services.forecast_engine import ForecastEngine
from backend.services.model_router import ModelRouter

def run_wape(sales, model):
    return ForecastEngine.run_backtest(pd.Series(sales), model)

def diagnose(label, sales, expected=None):
    non_zero = sales[sales > 0]
    if len(non_zero) == 0: print(f"{label}: all zeros"); return
    adi = len(sales) / len(non_zero)
    cv2 = (np.std(non_zero, ddof=1) / np.mean(non_zero))**2 if len(non_zero) > 1 else 0
    n = len(sales)
    acf91 = None
    if n >= 273:
        y = sales - sales.mean()
        d = np.sum(y**2)
        acf91 = round(float(np.sum(y[:-91]*y[91:])/d), 3) if d != 0 else 0.0
    wapes = {m: run_wape(sales, m) for m in ModelRouter.MODEL_REGISTRY}
    scores = {m: (None if w is None else round(max(0,(1-w))*100,1)) for m,w in wapes.items()}
    scorable = {m:s for m,s in scores.items() if s is not None}
    winner = max(scorable, key=scorable.get) if scorable else None
    icon = "OK" if (expected is None or winner == expected) else f"FAIL(exp={expected})"
    lines = [f"\n{label}: n={n}, ADI={adi:.2f}, CV2={cv2:.2f}, ACF91={acf91} => {winner} [{icon}]"]
    for m, s in sorted(scores.items(), key=lambda x: (x[1] or -999), reverse=True):
        w_str = f"{wapes[m]:.4f}" if wapes[m] is not None else "None"
        mk = " <-" if m == winner else ""
        lines.append(f"  {m:12s}: WAPE={w_str}, Score={s}{mk}")
    print("\n".join(lines))

RNG = np.random.default_rng(42)

# ===== SARIMAX: clean sin wave, no zeros =====
# No zeros so seasonal index is clean
# Phase: last 56 days should be maximum change (rising from trough to peak)
# With n=400, eval_start=344, 344%91=71 (trough) - PERFECT
for amp in [15, 20, 25, 30]:
    base = amp + 1  # min value = 1 (no clamp, no zeros)
    t = np.arange(400)
    raw = base + amp * np.sin(2*np.pi*t/91) + RNG.normal(0, 0.5, 400)
    sales = np.maximum(1, np.round(raw)).astype(float)
    diagnose(f"SKU-505 no-zero amp={amp} base={base} n=400", sales, "SARIMAX")

# ===== SARIMAX: 730 days, last 56 at trough =====
# 730 % 91 = 730 - 8*91 = 730-728 = 2. eval_start = 730-56=674; 674%91 = 674-7*91=674-637=37
# sin(2pi*37/91) = sin(2.558) = 0.558 -- rising phase. Not at trough.
# Try n=730 with phase offset so 674%91 = 68 (near trough=68.25)
# 730+X such that (730+X-56)%91=68: X = 68-37=31, so n=761
# Or: adjust seed so trough falls at correct position by varying n_days
for n_days in [400, 455, 491, 546]:
    es = n_days - 56
    phase = es % 91
    amp = 22; base = 23
    t = np.arange(n_days)
    raw = base + amp * np.sin(2*np.pi*t/91) + RNG.normal(0, 0.5, n_days)
    sales = np.maximum(1, np.round(raw)).astype(float)
    diagnose(f"SARIMAX n={n_days} eval_start%91={phase}", sales, "SARIMAX")

# ===== TSB: what makes TSB win? =====
# TSB = p*z. Wins when p is high (near-daily) but classical MA overforecasts.
# TSB updates p at EVERY step; SBA only at demand steps.
# TSB is better when the probability of demand is high but not 1.
# Need: ADI in [1.32, ...], CV2 < 0.49, TSB beats all others.
# From first diagnostic: Poisson(5.5) with std=0.57 gives Chronos-2 tie with ETS.
# TSB wins when series looks smooth-ish but with occasional zeros.
# Key insight: TSB's p tracks recent zeros well. When a zero occurs, TSB immediately
# reduces forecast; Chronos-2 still includes the pre-zero values in its 7-day window.
# Strategy: series with "bursts" separated by short gaps (2-3 zeros), then demand resumes.
# This is effectively the burst-gap pattern.
for gap_len, burst_len, size in [(2, 4, 5), (3, 5, 6), (2, 6, 4), (3, 4, 5)]:
    n = 365
    sales_t = np.zeros(n)
    i = 0
    in_burst = True
    count = 0
    while i < n:
        if in_burst:
            end = min(i+burst_len, n)
            sales_t[i:end] = np.maximum(1, RNG.poisson(size, end-i))
            i = end; count += 1
        else:
            end = min(i+gap_len, n)
            i = end
        in_burst = not in_burst
    diagnose(f"TSB burst/gap={burst_len}/{gap_len} size={size}", sales_t, "TSB")

# ETS: stable series with extremely low noise, slight trend
for std in [0.1, 0.3]:
    t = np.arange(365)
    raw = 20 + t * 0.01 + RNG.normal(0, std, 365)
    s = np.maximum(1, np.round(raw)).astype(float)
    diagnose(f"ETS trend std={std}", s, "ETS")
