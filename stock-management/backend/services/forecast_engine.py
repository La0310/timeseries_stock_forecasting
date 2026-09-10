"""
forecast_engine.py
Walk-forward backtesting (WAPE scoring) and probabilistic quantile generation.
Spec: 02_pipeline_workflow_v3.md §2.2, §2.2.1 / 03_addendum_spec_v3.md §1, §3.4

CRITICAL CONSTRAINTS (spec §2.2 / 03 §1):
  - _model_point_forecast is the ONLY per-model generator. It is called
    identically by run_backtest (one step at a time over expanding history
    prefixes) and by generate_quantiles (once at the requested horizon).
    No model has a separate backtest-only prediction function.
  - All six generators use only NumPy/pandas arithmetic.
    Do NOT import chronos-forecasting, lightgbm, statsmodels, or any ML lib.
"""

import numpy as np
import pandas as pd


class ForecastEngine:
    BACKTEST_WINDOWS = [14, 28, 56]  # expands until actual-demand sum > 0

    ETS_ALPHA = 0.3          # level smoothing (ETS, and base SARIMAX builds on)
    ETS_BETA = 0.1           # trend smoothing
    TSB_ALPHA = 0.2          # demand-probability smoothing
    TSB_BETA = 0.2           # demand-size smoothing
    SBA_ALPHA = 0.2          # Croston-style size/interval smoothing
    SEASONAL_PERIOD = 91     # quarterly, matches lag-91 seasonality test in §2.1

    # -------------------------------------------------------------------------
    # Shared helpers (used by SARIMAX generator only)
    # -------------------------------------------------------------------------

    @staticmethod
    def _holt_level_trend(
        sales: np.ndarray, alpha: float, beta: float
    ) -> tuple[float, float]:
        """Fixed-parameter Holt's linear method (no optimization/fitting).
        Returns the final (level, trend) after smoothing through the full
        series. Used by ETS directly, and by SARIMAX with a seasonal index
        layered on top.
        """
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
        partial data.
        """
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

    # -------------------------------------------------------------------------
    # THE single per-model deterministic generator
    # -------------------------------------------------------------------------

    @staticmethod
    def _model_point_forecast(
        history: pd.Series, model: str, horizon_days: int
    ) -> list[float]:
        """
        The ONLY per-model point-forecast generator.

        Called by:
          - run_backtest:       repeatedly at horizon_days=1 over expanding
                                history prefixes (walk-forward, no leakage)
          - generate_quantiles: once at the requested horizon_days

        No model may define a second backtest-only path (spec §2.2, 03 §1).
        All six models use only NumPy/pandas arithmetic — no ML packages.
        """
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
            # Level + trend (same as ETS) PLUS an additive seasonal index.
            # SARIMAX wins seasonal SKUs because this term exists, not via
            # a hardcoded demand-class mapping (spec §2.2 comment).
            level, trend = ForecastEngine._holt_level_trend(
                sales, ForecastEngine.ETS_ALPHA, ForecastEngine.ETS_BETA
            )
            seasonal = ForecastEngine._seasonal_index(
                sales, ForecastEngine.SEASONAL_PERIOD
            )
            return [
                max(
                    0.0,
                    level
                    + h * trend
                    + seasonal[(n + h - 1) % ForecastEngine.SEASONAL_PERIOD],
                )
                for h in range(1, horizon_days + 1)
            ]

        if model == "LightGBM":
            # Deterministic mock proxy for "tree ensembles handle extreme
            # spikes gracefully" (ModelRouter.REASONS["Erratic"]):
            # reacts to the most recent value and short-term trend rather
            # than smoothing them away.  Not a trained model of any kind.
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
            p, z = (
                (1.0, float(sales[has_sale][0])) if has_sale.any() else (0.0, 0.0)
            )
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
        # history fallback path (03_addendum_spec.md §4 item 4) — plain MA.
        window = sales[-7:] if n >= 7 else sales
        return [max(0.0, float(window.mean()))] * horizon_days

    # -------------------------------------------------------------------------
    # Backtest (walk-forward, leak-free)
    # -------------------------------------------------------------------------

    @staticmethod
    def run_backtest(history: pd.Series, model: str) -> float | None:
        """
        Walk-forward (expanding-origin) evaluation. For each timestep t
        inside the evaluation window, the forecast for t is generated from
        history strictly before t, using the exact same per-model generator
        (_model_point_forecast) that production forecasting uses. There is
        no separate backtest-only prediction function per model: a model's
        backtest score and its live forecast are one code path, so a leak
        can't be introduced into one without also breaking the other.
        Returns None if actual demand sums to zero across every window,
        including full history (Insufficient Data).

        Window expansion: 14 → 28 → 56 → n (spec §2.2 zero-guard).
        """
        sales = history.values
        n = len(sales)
        windows = [*ForecastEngine.BACKTEST_WINDOWS, n]
        for w in windows:
            w = min(w, n)
            eval_start = n - w
            actual = sales[eval_start:]
            if actual.sum() > 0:
                predicted = np.array(
                    [
                        ForecastEngine._model_point_forecast(
                            pd.Series(sales[: eval_start + i]), model, horizon_days=1
                        )[0]
                        for i in range(w)
                    ]
                )
                wape = float(np.sum(np.abs(actual - predicted)) / actual.sum())
                return wape
        return None  # Insufficient Data: zero actual demand across full history

    # -------------------------------------------------------------------------
    # Quantile generation
    # -------------------------------------------------------------------------

    @staticmethod
    def generate_quantiles(
        history: pd.Series, model: str, horizon_days: int, daily_std: float
    ) -> dict:
        """
        Returns {"P10": [...], "P50": [...], "P90": [...]} of length horizon_days.
        P50 comes from the model's deterministic generator; P10/P90 widen as
        sqrt(step) * daily_std around P50.

        daily_std is demand_analysis.daily_sales_std (spec §2.1), passed in
        by forecast_router.py — never recomputed here.
        """
        p50 = ForecastEngine._model_point_forecast(history, model, horizon_days)
        band = [1.65 * daily_std * np.sqrt(step + 1) for step in range(horizon_days)]
        p10 = [round(max(0.0, p - b), 1) for p, b in zip(p50, band)]
        p90 = [round(p + b, 1) for p, b in zip(p50, band)]
        return {"P10": p10, "P50": [round(p, 1) for p in p50], "P90": p90}
