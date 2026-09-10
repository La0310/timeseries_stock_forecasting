"""
demand_classifier.py
Syntetos-Boylan demand classification: ADI, CV², seasonality ACF.
Spec: 02_pipeline_workflow_v3.md §2.1 / 03_addendum_spec_v3.md §3.1
"""

import numpy as np
import pandas as pd


def autocorrelation(series: np.ndarray, lag: int) -> float | None:
    """Lag-k autocorrelation per spec §2.1 formula.
    Returns None (Insufficient History) when n < 3*lag — never guesses.
    """
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
    """Maps |ACF| magnitude to High / Medium / Low / Insufficient History."""
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
        """
        Computes demand classification metrics from a daily sales series.

        Returns a dict with:
          adi, cv_squared, zero_demand_ratio, daily_sales_std,
          demand_class, seasonality{weekly_acf, weekly_label,
                                     quarterly_acf, quarterly_label}

        IMPORTANT: daily_sales_std is computed over ALL days including zeros
        (ddof=1). It is deliberately different from CV²'s non-zero-only std.
        Both are computed here and only here — downstream modules consume
        daily_sales_std as passed through; they never recompute it.
        """
        sales = history_series.values
        n_periods = len(sales)
        non_zero = sales[sales > 0]

        # daily_sales_std: sample std over ALL days (including zeros).
        # Deliberately independent of the non-zero-only std used for CV²;
        # see spec §2.1 "Daily Sales Standard Deviation" note.
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
                    "weekly_acf": None,
                    "weekly_label": "Insufficient History",
                    "quarterly_acf": None,
                    "quarterly_label": "Insufficient History",
                },
            }

        adi = round(n_periods / len(non_zero), 2)
        cv_squared = (
            round(float((np.std(non_zero, ddof=1) / np.mean(non_zero)) ** 2), 2)
            if len(non_zero) > 1
            else 0.0
        )
        zero_ratio = round(float((n_periods - len(non_zero)) / n_periods), 2)

        # Classification rules per spec §2.1 (thresholds: ADI=1.32, CV²=0.49)
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
