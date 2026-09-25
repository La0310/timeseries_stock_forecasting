"""
forecast_engine.py
Walk-forward backtesting (WAPE scoring) and probabilistic quantile generation.
Spec: 02_pipeline_workflow_v3.md §2.2, §2.2.1 / 03_addendum_spec_v3.md §1, §3.4

REAL MODEL IMPLEMENTATIONS (Phase 2)
--------------------------------------
All six model generators now call real libraries:
  - LightGBM : lag-feature gradient-boosted regressor
  - SARIMAX  : statsmodels SARIMAX with AIC-selected orders
  - ETS      : statsmodels ETSModel (Holt-Winters)
  - TSB      : closed-form Teunter-Syntetos-Babai (no library needed)
  - SBA      : closed-form Syntetos-Boylan Approximation
  - Chronos-2: recency-weighted MA proxy (kept lightweight; swap for
               HuggingFace chronos-forecasting when GPU/RAM available)

LAZY MODEL CACHE
----------------
LightGBM, SARIMAX, and ETS are fitted objects that take seconds to train.
They are cached in the SQLite model_cache table as pickled blobs keyed by
  "{sku_id}|{model_name}|{history_hash}"
where history_hash = md5(history_bytes). If the history changes (new data),
the cache misses and the model is re-fitted automatically.

CRITICAL CONSTRAINTS (spec §2.2 / 03 §1):
  - _model_point_forecast is the ONLY per-model generator. It is called
    identically by run_backtest (one step at a time over expanding history
    prefixes) and by generate_quantiles (once at the requested horizon).
    No model has a separate backtest-only prediction function.
  - The lazy cache is consulted for production forecasts (generate_quantiles).
    During backtesting (run_backtest) the cache is intentionally bypassed
    because each backtest step uses a truncated history — caching truncated
    models would waste space and never be reused.
"""

from __future__ import annotations

import hashlib
import pickle
import threading
import warnings
from typing import Optional

import numpy as np
import pandas as pd

# LightGBM's C library is NOT thread-safe when fitted concurrently from
# Uvicorn's async threadpool. Serialise all lgb fit/predict calls.
_LGB_LOCK = threading.Lock()

# ---------------------------------------------------------------------------
# Optional library imports — degrade gracefully if not yet installed
# ---------------------------------------------------------------------------
try:
    import lightgbm as lgb
    _HAS_LGB = True
except ImportError:
    _HAS_LGB = False

try:
    from statsmodels.tsa.statespace.sarimax import SARIMAX as _SARIMAX
    from statsmodels.tsa.exponential_smoothing.ets import ETSModel as _ETS
    _HAS_SM = True
except ImportError:
    _HAS_SM = False


# ---------------------------------------------------------------------------
# Lag-feature builder for LightGBM
# ---------------------------------------------------------------------------

def _make_lag_features(sales: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Build a supervised (X, y) pair from a 1-D daily sales array.
    Features: lag-1,7,14,28, rolling-mean-7/14/28, day-of-week, month-index.
    Minimum 30 observations required; returns (None, None) if insufficient.
    """
    n = len(sales)
    if n < 30:
        return None, None

    lags = [1, 7, 14, 28]
    rows_X, rows_y = [], []

    for t in range(28, n):        # need at most 28-day lag
        feats = []
        for lag in lags:
            feats.append(sales[t - lag])
        feats.append(float(np.mean(sales[max(0, t - 7): t])))    # roll-7
        feats.append(float(np.mean(sales[max(0, t - 14): t])))   # roll-14
        feats.append(float(np.mean(sales[max(0, t - 28): t])))   # roll-28
        feats.append(float(t % 7))                                # day-of-week proxy
        feats.append(float((t // 30) % 12))                      # month proxy
        rows_X.append(feats)
        rows_y.append(sales[t])

    if len(rows_X) < 5:
        return None, None
    return np.array(rows_X, dtype=float), np.array(rows_y, dtype=float)


def _fit_lightgbm(sales: np.ndarray):
    """Fit and return a LightGBM regressor on lag features."""
    X, y = _make_lag_features(sales)
    if X is None:
        return None
    # Force C-contiguous float32 (LightGBM native dtype)
    X = np.ascontiguousarray(X, dtype=np.float32)
    y = np.ascontiguousarray(y, dtype=np.float32)
    model = lgb.LGBMRegressor(
        n_estimators=100, num_leaves=31, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, random_state=42, verbose=-1,
    )
    try:
        with _LGB_LOCK:
            model.fit(X, y)
    except (OSError, Exception):
        return None
    return model


def _predict_lightgbm(model, sales: np.ndarray, horizon: int) -> list[float]:
    """Multi-step recursive forecast with a fitted LightGBM model."""
    if model is None:
        return [max(0.0, float(np.mean(sales[-7:])))] * horizon

    history = list(sales.astype(float))
    preds = []
    for _ in range(horizon):
        n = len(history)
        feats = []
        for lag in [1, 7, 14, 28]:
            feats.append(history[-lag] if n >= lag else 0.0)
        feats.append(float(np.mean(history[-7:] if n >= 7 else history)))
        feats.append(float(np.mean(history[-14:] if n >= 14 else history)))
        feats.append(float(np.mean(history[-28:] if n >= 28 else history)))
        feats.append(float(n % 7))
        feats.append(float((n // 30) % 12))
        feat_arr = np.ascontiguousarray([feats], dtype=np.float32)
        with _LGB_LOCK:
            pred = float(model.predict(feat_arr)[0])
        pred = max(0.0, pred)
        preds.append(pred)
        history.append(pred)
    return preds


# ---------------------------------------------------------------------------
# SARIMAX helpers
# ---------------------------------------------------------------------------

_SARIMAX_ORDERS = [
    # (p, d, q) — search small grid, pick lowest AIC
    (1, 1, 1), (1, 0, 1), (0, 1, 1), (1, 1, 0), (0, 0, 1), (1, 0, 0),
]


def _fit_sarimax(sales: np.ndarray):
    """
    Fit SARIMAX with weekly seasonality (s=7).
    Iterates a small (p,d,q) grid and picks lowest AIC.
    Falls back to (1,0,0) if all fits fail.
    """
    n = len(sales)
    if n < 14:
        return None

    best_model, best_aic = None, np.inf
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # Try seasonal first
        for (p, d, q) in _SARIMAX_ORDERS[:3]:
            try:
                result = _SARIMAX(
                    sales, order=(p, d, q),
                    seasonal_order=(1, 0, 1, 7) if n >= 21 else (0, 0, 0, 0),
                    enforce_stationarity=False, enforce_invertibility=False,
                ).fit(disp=False, maxiter=50)
                if result.aic < best_aic:
                    best_aic, best_model = result.aic, result
            except Exception:
                pass
        # Fallback plain
        if best_model is None:
            try:
                best_model = _SARIMAX(sales, order=(1, 0, 0)).fit(disp=False, maxiter=30)
            except Exception:
                pass
    return best_model


def _predict_sarimax(model, horizon: int) -> list[float]:
    if model is None:
        return None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fc = model.forecast(steps=horizon)
        return [max(0.0, float(v)) for v in fc]
    except Exception:
        return None


# ---------------------------------------------------------------------------
# ETS helpers
# ---------------------------------------------------------------------------

def _fit_ets(sales: np.ndarray):
    """Fit Holt-Winters ETS (additive error, additive trend, additive seasonal)."""
    n = len(sales)
    if n < 14:
        return None

    # Require all non-negative values (ETS can't handle zeros with mult seasonal)
    y = np.maximum(sales, 0.01)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            seasonal_periods = 7 if n >= 21 else None
            error = "add"
            trend = "add" if n >= 14 else None
            seasonal = "add" if (seasonal_periods and n >= 2 * seasonal_periods) else None
            model = _ETS(
                y, error=error, trend=trend, seasonal=seasonal,
                seasonal_periods=seasonal_periods,
                initialization_method="estimated",
            ).fit(disp=False, maxiter=100)
            return model
        except Exception:
            try:
                # Ultra-simple fallback: SES only
                model = _ETS(y, error="add", trend=None, seasonal=None).fit(disp=False)
                return model
            except Exception:
                return None


def _predict_ets(model, horizon: int) -> list[float]:
    if model is None:
        return None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fc = model.forecast(horizon)
        return [max(0.0, float(v)) for v in fc]
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Closed-form helpers (unchanged from Phase 1 — no library needed)
# ---------------------------------------------------------------------------

def _holt_level_trend(
    sales: np.ndarray, alpha: float = 0.3, beta: float = 0.1
) -> tuple[float, float]:
    level, trend = float(sales[0]), 0.0
    for y in sales[1:]:
        prev_level = level
        level = alpha * y + (1 - alpha) * (level + trend)
        trend = beta * (level - prev_level) + (1 - beta) * trend
    return level, trend


def _seasonal_index(sales: np.ndarray, period: int) -> np.ndarray:
    if len(sales) < period:
        return np.zeros(period)
    mean = sales.mean()
    idx, counts = np.zeros(period), np.zeros(period)
    for t, y in enumerate(sales):
        pos = t % period
        idx[pos] += y - mean
        counts[pos] += 1
    counts[counts == 0] = 1
    return idx / counts


# ---------------------------------------------------------------------------
# Lazy model cache helpers
# ---------------------------------------------------------------------------

def _history_hash(history: pd.Series) -> str:
    """MD5 of the raw float bytes — changes when history changes."""
    return hashlib.md5(history.values.astype(np.float64).tobytes()).hexdigest()


def _cache_key(sku_id: str, model: str, h_hash: str) -> str:
    return f"{sku_id}|{model}|{h_hash}"


def _load_from_cache(db, cache_key: str):
    """Return unpickled model or None if not cached."""
    if db is None:
        return None
    try:
        from backend.db.models import ModelCache
        row = db.query(ModelCache).filter(ModelCache.cache_key == cache_key).first()
        if row:
            return pickle.loads(row.model_blob)
    except Exception:
        pass
    return None


def _save_to_cache(db, cache_key: str, model_obj, wape: Optional[float], h_hash: str) -> None:
    """Pickle and store model to SQLite model_cache."""
    if db is None or model_obj is None:
        return
    try:
        from datetime import datetime, timezone
        from backend.db.models import ModelCache
        blob = pickle.dumps(model_obj, protocol=pickle.HIGHEST_PROTOCOL)
        row = ModelCache(
            cache_key=cache_key,
            model_blob=blob,
            wape_score=wape,
            history_hash=h_hash,
            fitted_at=datetime.now(timezone.utc),
        )
        db.merge(row)   # INSERT or UPDATE
        db.commit()
    except Exception:
        pass            # cache failure is non-fatal


# ---------------------------------------------------------------------------
# THE single per-model deterministic generator
# ---------------------------------------------------------------------------

class ForecastEngine:
    BACKTEST_WINDOWS = [14, 28, 56]

    ETS_ALPHA = 0.3
    ETS_BETA  = 0.1
    TSB_ALPHA = 0.2
    TSB_BETA  = 0.2
    SBA_ALPHA = 0.2
    SEASONAL_PERIOD = 91

    @staticmethod
    def _model_point_forecast(
        history: pd.Series,
        model: str,
        horizon_days: int,
        *,
        db=None,
        sku_id: str = "",
        use_cache: bool = True,
    ) -> list[float]:
        """
        THE single per-model point-forecast generator.

        Parameters
        ----------
        history      : full sales history up to the forecast origin
        model        : model name string from ModelRouter.MODEL_REGISTRY
        horizon_days : number of steps to forecast
        db           : optional SQLAlchemy Session for lazy model caching
        sku_id       : SKU identifier (used as part of cache key)
        use_cache    : set False during backtesting to avoid caching partial histories

        Returns
        -------
        list[float] of length horizon_days (all values >= 0)
        """
        sales = history.values.astype(float)
        n = len(sales)
        if n == 0:
            return [0.0] * horizon_days

        # ------------------------------------------------------------------
        # LightGBM — lag-feature gradient-boosted regressor
        # ------------------------------------------------------------------
        if model == "LightGBM":
            if _HAS_LGB and n >= 30:
                h_hash = _history_hash(history)
                ck = _cache_key(sku_id, "LightGBM", h_hash)
                fitted = _load_from_cache(db, ck) if use_cache else None
                if fitted is None:
                    fitted = _fit_lightgbm(sales)
                    if use_cache:
                        _save_to_cache(db, ck, fitted, None, h_hash)
                preds = _predict_lightgbm(fitted, sales, horizon_days)
                return [max(0.0, p) for p in preds]
            else:
                # Fallback: recency-weighted mean (same as Chronos proxy)
                window = sales[-7:] if n >= 7 else sales
                weights = np.arange(1, len(window) + 1)
                level = float(np.dot(window, weights) / weights.sum())
                return [max(0.0, level)] * horizon_days

        # ------------------------------------------------------------------
        # SARIMAX — statsmodels state-space with weekly seasonality
        # ------------------------------------------------------------------
        if model == "SARIMAX":
            if _HAS_SM and n >= 14:
                h_hash = _history_hash(history)
                ck = _cache_key(sku_id, "SARIMAX", h_hash)
                fitted = _load_from_cache(db, ck) if use_cache else None
                if fitted is None:
                    fitted = _fit_sarimax(sales)
                    if use_cache:
                        _save_to_cache(db, ck, fitted, None, h_hash)
                preds = _predict_sarimax(fitted, horizon_days)
                if preds is not None:
                    return preds
            # Closed-form fallback (original Holt + seasonal index)
            level, trend = _holt_level_trend(sales, ForecastEngine.ETS_ALPHA, ForecastEngine.ETS_BETA)
            seasonal = _seasonal_index(sales, ForecastEngine.SEASONAL_PERIOD)
            return [
                max(0.0, level + h * trend + seasonal[(n + h - 1) % ForecastEngine.SEASONAL_PERIOD])
                for h in range(1, horizon_days + 1)
            ]

        # ------------------------------------------------------------------
        # ETS — statsmodels Holt-Winters ETS
        # ------------------------------------------------------------------
        if model == "ETS":
            if _HAS_SM and n >= 14:
                h_hash = _history_hash(history)
                ck = _cache_key(sku_id, "ETS", h_hash)
                fitted = _load_from_cache(db, ck) if use_cache else None
                if fitted is None:
                    fitted = _fit_ets(sales)
                    if use_cache:
                        _save_to_cache(db, ck, fitted, None, h_hash)
                preds = _predict_ets(fitted, horizon_days)
                if preds is not None:
                    return preds
            # Closed-form fallback: Holt's linear (original Phase 1 implementation)
            level, trend = _holt_level_trend(sales, ForecastEngine.ETS_ALPHA, ForecastEngine.ETS_BETA)
            return [max(0.0, level + h * trend) for h in range(1, horizon_days + 1)]

        # ------------------------------------------------------------------
        # Chronos-2 — recency-weighted moving average proxy
        # (swap for HuggingFace chronos-forecasting when GPU/large RAM available)
        # ------------------------------------------------------------------
        if model == "Chronos-2":
            window = sales[-7:] if n >= 7 else sales
            weights = np.arange(1, len(window) + 1)
            level = float(np.dot(window, weights) / weights.sum())
            return [max(0.0, level)] * horizon_days

        # ------------------------------------------------------------------
        # TSB — Teunter-Syntetos-Babai (closed-form, correct for intermittent)
        # ------------------------------------------------------------------
        if model == "TSB":
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

        # ------------------------------------------------------------------
        # SBA — Syntetos-Boylan Approximation (closed-form)
        # ------------------------------------------------------------------
        if model == "SBA":
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

        # Unrecognised model / None → plain moving-average fallback
        window = sales[-7:] if n >= 7 else sales
        return [max(0.0, float(window.mean()))] * horizon_days

    # -----------------------------------------------------------------------
    # Backtest (walk-forward, leak-free) — cache intentionally bypassed
    # -----------------------------------------------------------------------

    @staticmethod
    def run_backtest(history: pd.Series, model: str) -> float | None:
        """
        Walk-forward (expanding-origin) evaluation. Returns WAPE or None.

        The cache (db/sku_id) is NOT passed here: each backtest step uses a
        truncated history so caching would be wasteful and inaccurate.
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
                predicted = np.array([
                    ForecastEngine._model_point_forecast(
                        pd.Series(sales[: eval_start + i]),
                        model,
                        horizon_days=1,
                        use_cache=False,   # never cache truncated-history fits
                    )[0]
                    for i in range(w)
                ])
                wape = float(np.sum(np.abs(actual - predicted)) / actual.sum())
                return wape
        return None

    # -----------------------------------------------------------------------
    # Quantile generation — uses lazy cache for production forecasts
    # -----------------------------------------------------------------------

    @staticmethod
    def generate_quantiles(
        history: pd.Series,
        model: str,
        horizon_days: int,
        daily_std: float,
        *,
        db=None,
        sku_id: str = "",
    ) -> dict:
        """
        Returns {"P10": [...], "P50": [...], "P90": [...]} of length horizon_days.
        P50 is the model's point forecast (with lazy cache for fitted models).
        P10/P90 widen as sqrt(step) * daily_std around P50.
        """
        p50 = ForecastEngine._model_point_forecast(
            history, model, horizon_days,
            db=db, sku_id=sku_id, use_cache=True,
        )
        band = [1.65 * daily_std * np.sqrt(step + 1) for step in range(horizon_days)]
        p10  = [round(max(0.0, p - b), 1) for p, b in zip(p50, band)]
        p90  = [round(p + b, 1) for p, b in zip(p50, band)]
        return {"P10": p10, "P50": [round(p, 1) for p in p50], "P90": p90}
