# Pipeline Architecture & Implementation Runbook (v3)

## 1. Purpose & Pipeline Boundaries

This document details the modular service pipeline powering `POST /api/forecast` and `POST /api/purchase-order`. The demo uses deterministic mock engines for heavy computational tasks (model inference) while running real mathematical formulas for demand classification (ADI, CV², seasonality), leak-free backtest-driven model scoring, and safety stock calculations.

Every field name below is the literal key used in the API response in §3. `demand_classifier.py`, `model_router.py`, `forecast_engine.py`, `inventory_recommender.py`, and `purchase_order_service.py` must be implemented as separate files; none of them may be collapsed into `forecast_router.py` or `purchase_order_router.py`.

---

## 2. Core Service Modules (`backend/services/`)

### 2.1 Demand Classifier (`demand_classifier.py`)

Computes empirical diagnostic metrics from historical sales, split into two independent signals: an intermittency/volatility class (four buckets, mutually exclusive) and a seasonality strength (continuous, orthogonal to the class). A SKU can be, for example, both "Smooth" and "highly seasonal" at once; these are not competing labels.

**Average Demand Interval (ADI):**
$$ADI = \frac{N}{\sum_{t=1}^N \mathbb{I}(y_t > 0)}$$

**Squared Coefficient of Variation (CV²):**
$$CV^2 = \left( \frac{\sigma_{y>0}}{\mu_{y>0}} \right)^2$$

**Zero-Demand Ratio:**
$$\text{Zero Ratio} = \frac{\sum_{t=1}^N \mathbb{I}(y_t = 0)}{N}$$

**Daily Sales Standard Deviation (clarification added):**
$$\sigma_{\text{daily}} = \operatorname{std}(y_1, \dots, y_N), \quad \text{sample std (ddof=1), over all } N \text{ days, including zero-demand days}$$

This is a different statistic from `CV²`'s $\sigma_{y>0}$ above, and the difference matters downstream. `CV²` measures order-size volatility conditional on a sale occurring, and exists solely to classify intermittency (Smooth / Intermittent / Erratic / Lumpy). `daily_sales_std` measures unconditional day-to-day demand variability, the quantity that both the P10/P90 forecast band (§2.2) and the Safety Stock formula (§2.4, `03_addendum_spec.md` §3.2) are meant to buffer against. `DemandClassifier.analyze()` is the single place both are computed; `forecast_engine.py` and `inventory_recommender.py` consume `demand_analysis.daily_sales_std` as passed through by `forecast_router.py`, they do not recompute it and do not substitute `cv_squared`'s intermediate non-zero std for it.

**Classification rules (ADI threshold 1.32, CV² threshold 0.49):**

- `Smooth`: ADI < 1.32 and CV² < 0.49
- `Intermittent`: ADI ≥ 1.32 and CV² < 0.49
- `Erratic`: ADI < 1.32 and CV² ≥ 0.49
- `Lumpy`: ADI ≥ 1.32 and CV² ≥ 0.49

**Seasonality (lag-k autocorrelation):**
$$\rho_k = \frac{\sum_{t=1}^{N-k}(y_t - \bar y)(y_{t+k} - \bar y)}{\sum_{t=1}^{N}(y_t - \bar y)^2}$$

Computed at lag 7 (weekly) and lag 91 (quarterly). A lag is only estimated if the series has at least 3× that many periods of history; otherwise it is reported as `null` ("Insufficient History"), never guessed or defaulted to zero.

| \|ρ\| | Label |
|---|---|
| ≥ 0.6 | High |
| 0.3 - 0.6 | Medium |
| < 0.3 | Low |

```python
import numpy as np
import pandas as pd


def autocorrelation(series: np.ndarray, lag: int) -> float | None:
    n = len(series)
    if n < 3 * lag:
        return None  # insufficient history for a reliable estimate
    y = series - series.mean()
    denominator = np.sum(y ** 2)
    if denominator == 0:
        return 0.0
    numerator = np.sum(y[: n - lag] * y[lag:])
    return round(float(numerator / denominator), 2)


def label_seasonality(acf: float | None) -> str:
    if acf is None:
        return "Insufficient History"
    magnitude = abs(acf)
    if magnitude >= 0.6:
        return "High"
    if magnitude >= 0.3:
        return "Medium"
    return "Low"


class DemandClassifier:
    @staticmethod
    def analyze(history_series: pd.Series) -> dict:
        sales = history_series.values
        n_periods = len(sales)
        non_zero = sales[sales > 0]

        # daily_sales_std: sample std over ALL days (including zeros).
        # Deliberately independent of the non-zero-only std used for CV²
        # below; see the "Daily Sales Standard Deviation" note above.
        daily_sales_std = (
            round(float(np.std(sales, ddof=1)), 2) if n_periods > 1 else 0.0
        )

        if len(non_zero) == 0:
            return {
                "adi": float("inf"),
                "cv_squared": 0.0,
                "zero_demand_ratio": 1.0,
                "daily_sales_std": daily_sales_std,
                "demand_class": "Zero Demand",
                "seasonality": {
                    "weekly_acf": None, "weekly_label": "Insufficient History",
                    "quarterly_acf": None, "quarterly_label": "Insufficient History",
                },
            }

        adi = round(n_periods / len(non_zero), 2)
        cv_squared = (
            round(float((np.std(non_zero, ddof=1) / np.mean(non_zero)) ** 2), 2)
            if len(non_zero) > 1 else 0.0
        )
        zero_ratio = round(float((n_periods - len(non_zero)) / n_periods), 2)

        if adi < 1.32 and cv_squared < 0.49:
            demand_class = "Smooth"
        elif adi >= 1.32 and cv_squared < 0.49:
            demand_class = "Intermittent"
        elif adi < 1.32 and cv_squared >= 0.49:
            demand_class = "Erratic"
        else:
            demand_class = "Lumpy"

        weekly_acf = autocorrelation(sales, lag=7)
        quarterly_acf = autocorrelation(sales, lag=91)

        return {
            "adi": adi,
            "cv_squared": cv_squared,
            "zero_demand_ratio": zero_ratio,
            "daily_sales_std": daily_sales_std,
            "demand_class": demand_class,
            "seasonality": {
                "weekly_acf": weekly_acf,
                "weekly_label": label_seasonality(weekly_acf),
                "quarterly_acf": quarterly_acf,
                "quarterly_label": label_seasonality(quarterly_acf),
            },
        }
```

Note: the 1.32 / 0.49 cutoffs above are calibrated for this build's curated demo catalog. See `03_addendum_spec.md` §5 for a known limitation encountered when the same thresholds were pointed at a larger, real SKU catalog.

---

### 2.2 Forecast Engine (`forecast_engine.py`)

Owns the two things `model_router.py` and the API response depend on: rolling backtests (for WAPE scoring) and the final quantile forecast for the selected model.

**Backtest methodology (walk-forward, leak-free by construction):** a prior draft of this module let a per-model backtest-prediction function receive the actual values of the window it was being scored against. That is an open door to leakage: nothing stops an implementation from returning something like `actual * 0.95 + small_term`, which scores near-perfectly regardless of whether the model fits the demand pattern at all, and would structurally favor whichever model's leak coefficient sits closest to 1.0, independent of demand pattern.

v3 closes this at the interface level rather than trusting every model implementation to behave correctly: there is no separate backtest-only prediction function. Backtesting instead walks forward through the evaluation window one step at a time, and at each step calls the exact same `_model_point_forecast()` generator that production forecasting uses, using only history strictly before that step. A model's backtest score and its live forecast are now produced by one code path; a leak introduced into one is necessarily a leak in the other, and would be caught the same way an ordinary forecast error would be.

**Backtest windowing (zero-guard):** WAPE requires Σy > 0 in the evaluation window. Start with a 14-period rolling window; if the actual demand sums to zero, double the window (28, then 56) up to the full available history. If the sum is still zero at full history, the model cannot be scored: return `None`, not a divide-by-zero or a fabricated number.

$$\text{WAPE}_m = \frac{\sum_t |y_t - \hat y_{m,t}|}{\sum_t y_t} \quad \text{(only where } \sum_t y_t > 0\text{)}$$

**Quantile generation:** P50 is the model-specific deterministic point forecast, never randomly sampled. The P10/P90 spread widens with the square root of the forecast step (standard treatment of compounding uncertainty over a horizon), scaled by the SKU's `daily_sales_std` (defined in §2.1; sourced from `demand_classifier.py`, not recomputed here).

```python
class ForecastEngine:
    BACKTEST_WINDOWS = [14, 28, 56]  # expands until actual-demand sum > 0

    @staticmethod
    def run_backtest(history: pd.Series, model: str) -> float | None:
        """Walk-forward (expanding-origin) evaluation. For each timestep t
        inside the evaluation window, the forecast for t is generated from
        history strictly before t, using the exact same per-model generator
        (_model_point_forecast) that production forecasting uses. There is
        no separate backtest-only prediction function per model: a model's
        backtest score and its live forecast are one code path, so a leak
        can't be introduced into one without also breaking the other.
        Returns None if actual demand sums to zero across every window,
        including full history (Insufficient Data)."""
        sales = history.values
        n = len(sales)
        windows = [*ForecastEngine.BACKTEST_WINDOWS, n]
        for w in windows:
            w = min(w, n)
            eval_start = n - w
            actual = sales[eval_start:]
            if actual.sum() > 0:
                predicted = np.array([
                    ForecastEngine._model_point_forecast(
                        pd.Series(sales[: eval_start + i]), model, horizon_days=1
                    )[0]
                    for i in range(w)
                ])
                wape = float(np.sum(np.abs(actual - predicted)) / actual.sum())
                return wape
        return None  # Insufficient Data: zero actual demand across full history

    @staticmethod
    def generate_quantiles(
        history: pd.Series, model: str, horizon_days: int, daily_std: float
    ) -> dict:
        """Returns {"P10": [...], "P50": [...], "P90": [...]} of length horizon_days.
        P50 comes from the model's deterministic generator; P10/P90 widen as
        sqrt(step) * daily_std around P50. daily_std is demand_analysis.daily_sales_std
        (§2.1), passed in by forecast_router.py, never recomputed here."""
        p50 = ForecastEngine._model_point_forecast(history, model, horizon_days)
        band = [1.65 * daily_std * np.sqrt(step + 1) for step in range(horizon_days)]
        p10 = [round(max(0.0, p - b), 1) for p, b in zip(p50, band)]
        p90 = [round(p + b, 1) for p, b in zip(p50, band)]
        return {"P10": p10, "P50": [round(p, 1) for p in p50], "P90": p90}

    # _model_point_forecast: the ONLY per-model deterministic generator, one
    # per MODEL_REGISTRY entry, used identically by run_backtest (called
    # repeatedly at horizon_days=1 over expanding prefixes) and by
    # generate_quantiles (called once at the requested horizon_days). No
    # model may define a second, backtest-only prediction path. SARIMAX's
    # generator must genuinely incorporate a seasonal component so that it
    # wins the WAPE comparison on seasonal SKUs on backtest merit, not via
    # a hardcoded pattern-to-model mapping. Routing stays purely
    # score-driven (see §2.3). Concrete per-model logic is in §2.2.1.
```

This raises backtest cost from O(1) to O(w) calls to `_model_point_forecast` per model per SKU (w up to 56, or full history). At demo data volumes this remains comfortably inside the <500ms mock-latency ceiling (`03_addendum_spec.md` §4 item 1). A production catalog at full scale would need to cache or vectorize the per-step forecasts rather than recomputing sequentially; that is out of scope for this build.

#### 2.2.1 Per-Model Deterministic Generators (`_model_point_forecast`, clarification added)

**All six generators below are lightweight deterministic formulas named after real forecasting methods for narrative and UI flavor. None of them load a pretrained checkpoint, call an external inference API, train a model at request time, or depend on the actual `chronos-forecasting` or `lightgbm` packages.** Implementing any of them by installing and running the real library would blow the <500ms latency ceiling (`03_addendum_spec.md` §4 item 1) and add a dependency this build does not need (see `03_addendum_spec.md` §4 item 9). TSB and SBA below are the standard textbook intermittent-demand formulas; ETS and SARIMAX are standard smoothing formulas; LightGBM and Chronos-2 are deterministic stand-ins chosen to be qualitatively distinctive (spike-reactive vs. flat-stable), not attempts to imitate the real libraries' internals.

Fixed smoothing constants below are hardcoded, not fitted, consistent with this being a demo mock engine rather than a trained model.

```python
class ForecastEngine:
    # ... BACKTEST_WINDOWS, run_backtest, generate_quantiles as above ...

    ETS_ALPHA = 0.3          # level smoothing (ETS, and the level/trend base SARIMAX builds on)
    ETS_BETA = 0.1           # trend smoothing
    TSB_ALPHA = 0.2          # demand-probability smoothing
    TSB_BETA = 0.2           # demand-size smoothing
    SBA_ALPHA = 0.2          # Croston-style size/interval smoothing
    SEASONAL_PERIOD = 91     # quarterly, matches the lag-91 seasonality test in §2.1

    @staticmethod
    def _holt_level_trend(sales: np.ndarray, alpha: float, beta: float) -> tuple[float, float]:
        """Fixed-parameter Holt's linear method (no optimization/fitting).
        Returns the final (level, trend) after smoothing through the full
        series. Used by ETS directly, and by SARIMAX with a seasonal index
        layered on top."""
        level, trend = float(sales[0]), 0.0
        for y in sales[1:]:
            prev_level = level
            level = alpha * y + (1 - alpha) * (level + trend)
            trend = beta * (level - prev_level) + (1 - beta) * trend
        return level, trend

    @staticmethod
    def _seasonal_index(sales: np.ndarray, period: int) -> np.ndarray:
        """Mean deviation-from-series-mean at each position in the cycle.
        Returns an all-zero index (no seasonal adjustment) until history
        covers at least one full period, rather than guessing a cycle from
        partial data."""
        if len(sales) < period:
            return np.zeros(period)
        mean = sales.mean()
        idx = np.zeros(period)
        counts = np.zeros(period)
        for t, y in enumerate(sales):
            pos = t % period
            idx[pos] += y - mean
            counts[pos] += 1
        counts[counts == 0] = 1
        return idx / counts

    @staticmethod
    def _model_point_forecast(history: pd.Series, model: str, horizon_days: int) -> list[float]:
        sales = history.values.astype(float)
        n = len(sales)
        if n == 0:
            return [0.0] * horizon_days

        if model == "ETS":
            # Standard Holt's linear method: stable baseline, no seasonal term.
            level, trend = ForecastEngine._holt_level_trend(
                sales, ForecastEngine.ETS_ALPHA, ForecastEngine.ETS_BETA
            )
            return [max(0.0, level + h * trend) for h in range(1, horizon_days + 1)]

        if model == "SARIMAX":
            # Level + trend, same as ETS, PLUS an additive seasonal index.
            # This is the "genuinely incorporate a seasonal component" the
            # comment in §2.2 requires: SARIMAX only wins SKU-505's backtest
            # because this term exists, not because of a hardcoded mapping.
            level, trend = ForecastEngine._holt_level_trend(
                sales, ForecastEngine.ETS_ALPHA, ForecastEngine.ETS_BETA
            )
            seasonal = ForecastEngine._seasonal_index(sales, ForecastEngine.SEASONAL_PERIOD)
            return [
                max(0.0, level + h * trend + seasonal[(n + h - 1) % ForecastEngine.SEASONAL_PERIOD])
                for h in range(1, horizon_days + 1)
            ]

        if model == "LightGBM":
            # Deterministic mock proxy for "tree ensembles handle extreme
            # spikes gracefully" (model_router.py REASONS["Erratic"]):
            # reacts to the most recent value and short-term trend rather
            # than smoothing them away. Not a trained model of any kind.
            last = sales[-1]
            mean_7 = sales[-7:].mean() if n >= 7 else sales.mean()
            slope = (sales[-1] - sales[-7]) / 7 if n >= 7 else 0.0
            return [
                max(0.0, 0.4 * last + 0.3 * mean_7 + 0.3 * max(0.0, slope) * h)
                for h in range(1, horizon_days + 1)
            ]

        if model == "Chronos-2":
            # Deterministic mock proxy for "continuous non-zero sales
            # frequency, low transaction variance": a recency-weighted
            # moving average, flat across the horizon. Not a zero-shot
            # foundation-model call of any kind.
            window = sales[-7:] if n >= 7 else sales
            weights = np.arange(1, len(window) + 1)
            level = float(np.dot(window, weights) / weights.sum())
            return [max(0.0, level)] * horizon_days

        if model == "TSB":
            # Teunter-Syntetos-Babai: smoothed demand-occurrence probability
            # p and smoothed demand-size-given-sale z. Forecast = p * z,
            # constant across the horizon (standard intermittent-demand
            # treatment; no trend term).
            has_sale = sales > 0
            p, z = (1.0, float(sales[has_sale][0])) if has_sale.any() else (0.0, 0.0)
            for y in sales:
                if y > 0:
                    p = p + ForecastEngine.TSB_ALPHA * (1 - p)
                    z = z + ForecastEngine.TSB_BETA * (y - z)
                else:
                    p = p * (1 - ForecastEngine.TSB_ALPHA)
            return [max(0.0, p * z)] * horizon_days

        if model == "SBA":
            # Syntetos-Boylan Approximation: Croston's method (size z and
            # inter-demand interval q, each smoothed only at demand
            # occurrences) with the standard bias correction (1 - alpha/2).
            nonzero_idx = np.flatnonzero(sales > 0)
            if len(nonzero_idx) == 0:
                return [0.0] * horizon_days
            z = float(sales[nonzero_idx[0]])
            q = float(nonzero_idx[0] + 1)
            last_idx = nonzero_idx[0]
            for idx in nonzero_idx[1:]:
                interval = idx - last_idx
                z = z + ForecastEngine.SBA_ALPHA * (sales[idx] - z)
                q = q + ForecastEngine.SBA_ALPHA * (interval - q)
                last_idx = idx
            bias_correction = 1 - ForecastEngine.SBA_ALPHA / 2
            forecast = bias_correction * (z / q) if q > 0 else 0.0
            return [max(0.0, forecast)] * horizon_days

        # model is None or unrecognized: the Insufficient-Data / limited-
        # history fallback path (`03_addendum_spec.md` §4 item 4), a plain
        # moving-average baseline.
        window = sales[-7:] if n >= 7 else sales
        return [max(0.0, float(window.mean()))] * horizon_days
```

---

### 2.3 Model Router (`model_router.py`)

Ranks candidate forecasting algorithms by backtest WAPE. Selection is always score-driven; there is no hardcoded demand-class-to-model mapping, that would be exactly the invisible-black-box behavior the product spec prohibits. This module consumes only the leak-free WAPE scores `ForecastEngine.run_backtest()` now guarantees (§2.2); the routing logic itself did not need to change, the bug lived entirely upstream in how those scores were produced.

$$\text{Score}_m = \begin{cases} \max\left(0, \left(1 - \text{WAPE}_m\right) \times 100\right) & \text{if WAPE}_m \text{ is defined} \\ \text{null (Insufficient Data)} & \text{otherwise} \end{cases}$$

```python
class ModelRouter:
    MODEL_REGISTRY = ["Chronos-2", "LightGBM", "SARIMAX", "ETS", "TSB", "SBA"]

    REASONS = {
        "Smooth": "Continuous demand cadence with stable variance; neural/deep models capture non-linear trends.",
        "Intermittent": "Sporadic demand with long zero runs; TSB (Croston-family) separates order probability from volume.",
        "Erratic": "Frequent transactions with volatile basket sizes; tree ensembles handle extreme spikes gracefully.",
        "Lumpy": "Sparse cadence with highly variable order quantities; SBA (bias-corrected bootstrap) prevents over-replenishment.",
    }

    @staticmethod
    def route(demand_profile: dict, backtest_wape: dict) -> dict:
        scores = {}
        for model, wape in backtest_wape.items():
            scores[model] = None if wape is None else round(max(0.0, (1.0 - wape) * 100), 1)

        scorable = {m: s for m, s in scores.items() if s is not None}
        if not scorable:
            return {
                "selected_model": None,
                "model_scores": scores,
                "justification": "Insufficient sales history to backtest any candidate; falling back to moving-average baseline.",
            }

        selected_model = max(scorable, key=scorable.get)
        return {
            "selected_model": selected_model,
            "model_scores": scores,
            "justification": ModelRouter.REASONS.get(
                demand_profile["demand_class"], "Optimal historical backtest accuracy."
            ),
        }
```

---

### 2.4 Inventory Recommender (`inventory_recommender.py`)

Applies category-differentiated safety stock and Inventory-Position-gated reorder logic, and is the single source of the `status` field consumed by Screens 1, 2, and 3.

**Lead Time Demand:** $\hat D_L = \sum_{t=1}^{L} \hat y_t$

**Category-Differentiated Safety Stock:** $SS = \lceil Z_{\text{category}} \times \sigma_d \times \sqrt{L} \rceil$. `Z_category` is looked up per SKU Category (`03_addendum_spec.md` §3.2); any Category not in the mapping uses the 95% / Z=1.65 default. $\sigma_d$ is `daily_sales_std` from `demand_classifier.py` (§2.1), passed through unmodified, not recomputed here.

**Reorder Point:** $ROP = \hat D_L + SS$

**Inventory Position:** $IP = \text{OnHand} + \text{OnOrder}$, sourced from `purchase_order_service.py` (§2.5). No Backorders term: see `03_addendum_spec.md` §6, item 2, for why it was dropped rather than approximated.

**Net Recommended Order (Inventory-Position-gated):**
$$Q_{\text{raw}} = \begin{cases} 0 & IP > ROP \\ \max\left(0, \left(\sum_{t=1}^{\text{horizon\_days}} \hat y_t + SS\right) - IP\right) & IP \le ROP \end{cases}$$
$$Q_{\text{final}} = \left\lceil \frac{Q_{\text{raw}}}{\text{Pack Size}} \right\rceil \times \text{Pack Size}$$

`horizon_days` defaults to 28, matching the API request payload.

**Status (single-basis, matches `01_product_spec.md` §3):**

- `Critical` if Inventory Position ≤ ROP
- `Low` if ROP < Inventory Position ≤ 1.5 × ROP
- `Healthy` if Inventory Position > 1.5 × ROP

```python
import numpy as np

SERVICE_LEVEL_Z = {
    "Beverage": 1.65,          # 95% - fast-turning, short lead time, low stockout cost
    "Office": 1.65,            # 95% - fast-turning consumables, low stockout cost
    "Hardware": 1.96,          # 97.5% - higher unit cost, promo-driven demand spikes
    "MRO / Parts": 2.33,       # 99% - long lead time; stockout can stall the buyer's own operations
    "Seasonal overlay": 1.28,  # 90% - overstock/markdown risk outweighs stockout risk here
}
DEFAULT_Z = 1.65  # any Category not listed above falls back to the 95% baseline


class InventoryRecommender:
    @staticmethod
    def service_level_z(category: str) -> float:
        return SERVICE_LEVEL_Z.get(category, DEFAULT_Z)

    @staticmethod
    def classify_status(inventory_position: float, reorder_point: float) -> str:
        if inventory_position <= reorder_point:
            return "Critical"
        if inventory_position <= 1.5 * reorder_point:
            return "Low"
        return "Healthy"

    @staticmethod
    def calculate(
        current_stock: int,
        on_order_stock: int,
        category: str,
        lead_time_days: int,
        daily_sales_std: float,
        daily_forecast: list,
        pack_size: int = 1,
    ) -> dict:
        z_score = InventoryRecommender.service_level_z(category)
        inventory_position = current_stock + on_order_stock

        lead_demand = float(np.sum(daily_forecast[:lead_time_days]))
        total_horizon_demand = float(np.sum(daily_forecast))

        safety_stock = int(np.ceil(z_score * daily_sales_std * np.sqrt(lead_time_days)))
        reorder_point = int(np.ceil(lead_demand + safety_stock))

        if inventory_position > reorder_point:
            unrounded_order = 0  # IP-gate: lead-time buffer is already covered
        else:
            unrounded_order = max(
                0, int(np.ceil((total_horizon_demand + safety_stock) - inventory_position))
            )

        if pack_size > 1 and unrounded_order > 0:
            cases_needed = int(np.ceil(unrounded_order / pack_size))
            recommended_order = cases_needed * pack_size
        else:
            cases_needed = unrounded_order
            recommended_order = unrounded_order

        return {
            "current_stock": current_stock,
            "on_order_stock": on_order_stock,
            "inventory_position": inventory_position,
            "lead_time_days": lead_time_days,
            "lead_time_demand": round(lead_demand, 1),
            "service_level_z": z_score,
            "safety_stock": safety_stock,
            "reorder_point": reorder_point,
            "unrounded_order": unrounded_order,
            "pack_size": pack_size,
            "recommended_cases": cases_needed,
            "recommended_order": recommended_order,
            "status": InventoryRecommender.classify_status(inventory_position, reorder_point),
        }
```

---

### 2.5 Purchase Order Service (`purchase_order_service.py`, New in v3)

Owns the on-order ledger that `inventory_recommender.py`'s Inventory Position calculation reads from. A demo-scope, in-memory store; see `01_product_spec.md` Guardrail 5 for the persistence caveat.

```python
"""In-memory on-order ledger for the demo. A production build would back
this with a real purchase-order table; the interface is written so that
swap is a storage-layer change only, not a call-site change."""

from collections import defaultdict


class PurchaseOrderService:
    _on_order_by_sku: dict[str, int] = defaultdict(int)
    _po_sequence: int = 0

    @classmethod
    def record_purchase_order(cls, sku_id: str, quantity: int) -> dict:
        cls._po_sequence += 1
        po_id = f"PO-2026-{cls._po_sequence:06d}"
        cls._on_order_by_sku[sku_id] += quantity
        return {
            "po_id": po_id,
            "sku_id": sku_id,
            "quantity_ordered": quantity,
            "on_order_stock": cls._on_order_by_sku[sku_id],
        }

    @classmethod
    def get_on_order_stock(cls, sku_id: str) -> int:
        return cls._on_order_by_sku[sku_id]
```

---

## 3. API Specification

### 3.1 Endpoint: `POST /api/forecast`

**Request Payload:**

```json
{
  "sku_id": "SKU-101",
  "horizon_days": 28
}
```

**Response Payload** (field names here are the literal keys every service above returns; `forecast_router.py` must not rename them):

```json
{
  "sku_id": "SKU-101",
  "limited_history": false,
  "product_name": "Coca-Cola 330ml Can",
  "category": "Beverage",
  "demand_analysis": {
    "adi": 1.02,
    "cv_squared": 0.18,
    "zero_demand_ratio": 0.02,
    "daily_sales_std": 2.5,
    "demand_class": "Smooth",
    "seasonality": {
      "weekly_acf": 0.12,
      "weekly_label": "Low",
      "quarterly_acf": null,
      "quarterly_label": "Insufficient History"
    }
  },
  "routing": {
    "selected_model": "Chronos-2",
    "model_scores": {
      "Chronos-2": 91.7,
      "LightGBM": 88.4,
      "SARIMAX": 82.3,
      "ETS": 79.1,
      "TSB": 64.0,
      "SBA": 58.2
    },
    "justification": "Continuous demand cadence with stable variance; neural/deep models capture non-linear trends."
  },
  "forecast": {
    "summary": {
      "7_days": 38,
      "14_days": 76,
      "28_days": 152
    },
    "quantiles": {
      "P10": [1.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
      "P50": [5.3, 5.4, 5.4, 5.4, 5.5, 5.6, 5.6],
      "P90": [9.4, 11.2, 12.5, 13.7, 14.7, 15.7, 16.5]
    }
  },
  "replenishment": {
    "current_stock": 40,
    "on_order_stock": 0,
    "inventory_position": 40,
    "lead_time_days": 7,
    "lead_time_demand": 38.0,
    "service_level_z": 1.65,
    "safety_stock": 11,
    "reorder_point": 49,
    "pack_size": 24,
    "recommended_cases": 6,
    "recommended_order": 144,
    "status": "Critical"
  }
}
```

The wide, quickly-zero-clamped P10 band above is the formula working correctly, not a mistake: at a modest ~5.4 units/day, day-to-day sales counts behave close to Poisson, so `daily_sales_std` (≈2.5) is naturally a large fraction of the mean, and the 1.65×σ×√step band swallows P10 by day 2. A higher-volume SKU produces a proportionally tighter band; this is not specific to SKU-101.

New in v3: `on_order_stock`, `inventory_position`, and `service_level_z` inside `replenishment`. `current_stock` keeps its v2 meaning (raw on-hand); it no longer solely drives `status` or the order gate, `inventory_position` does. New in this clarification pass: `daily_sales_std` inside `demand_analysis`, the single computed value that `forecast.quantiles` (§2.2) and `replenishment.safety_stock` (§2.4) both read, so the two panels can never quietly disagree about which day-to-day variability figure they're buffering against. New in this patch: `limited_history`, a top-level boolean set whenever fewer than 14 historical records exist or every candidate model scores `null` (`03_addendum_spec.md` §4 item 4); the frontend banner (Guardrail 3, `01_product_spec.md` §4) reads this flag directly rather than inferring it from array lengths or `null` scores itself.

Note what remains gone from the original schema: `stockout_risk` as a separate binary. It duplicated `status` and could drift out of sync with it, that duplication is exactly how the Screen 1 badge contradiction happened in v1. `status` is the single field carrying that meaning end to end.

### 3.2 Endpoint: `POST /api/purchase-order` (New in v3)

**Request Payload:**

```json
{
  "sku_id": "SKU-101",
  "quantity": 144
}
```

**Response Payload:**

```json
{
  "po_id": "PO-2026-000001",
  "sku_id": "SKU-101",
  "quantity_ordered": 144,
  "current_stock": 40,
  "on_order_stock": 144,
  "inventory_position": 184,
  "reorder_point": 49,
  "new_status": "Healthy"
}
```

`new_status` is computed by the same `InventoryRecommender.classify_status()` used everywhere else, called with the post-PO Inventory Position. Screen 3 (`01_product_spec.md` §3) renders `new_status` verbatim; it must not assume or hardcode any particular before/after transition (Guardrail 6, `01_product_spec.md` §4).

Field-level rationale for what changed since v2 is in `03_addendum_spec.md` §6 (Revision Log v2 → v3).
