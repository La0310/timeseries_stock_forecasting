"""
model_router.py
WAPE-score-driven model selection. No hardcoded demand-class→model mapping.
Spec: 02_pipeline_workflow_v3.md §2.3 / 03_addendum_spec_v3.md §3.4
"""


class ModelRouter:
    MODEL_REGISTRY = ["Chronos-2", "LightGBM", "SARIMAX", "ETS", "TSB", "SBA"]

    # Narrative justifications per demand class — for UI display only.
    # Selection is always determined by backtest WAPE, never by this dict.
    REASONS = {
        "Smooth": (
            "Continuous demand cadence with stable variance; "
            "neural/deep models capture non-linear trends."
        ),
        "Intermittent": (
            "Sporadic demand with long zero runs; TSB (Croston-family) "
            "separates order probability from volume."
        ),
        "Erratic": (
            "Frequent transactions with volatile basket sizes; "
            "tree ensembles handle extreme spikes gracefully."
        ),
        "Lumpy": (
            "Sparse cadence with highly variable order quantities; "
            "SBA (bias-corrected bootstrap) prevents over-replenishment."
        ),
    }

    @staticmethod
    def route(demand_profile: dict, backtest_wape: dict) -> dict:
        """
        Converts WAPE values to scores, picks the highest scorer.

        Score_m = max(0, (1 - WAPE_m) * 100) if WAPE_m is defined, else null.
        If every model scores null (all-zero demand history), selected_model
        is None and the caller must set limited_history=True.

        Parameters
        ----------
        demand_profile : dict
            Output of DemandClassifier.analyze() — only demand_class is used
            here (to look up the REASONS justification string).
        backtest_wape : dict
            {model_name: wape_float_or_None} from ForecastEngine.run_backtest()

        Returns
        -------
        dict with selected_model, model_scores, justification
        """
        scores = {}
        for model, wape in backtest_wape.items():
            scores[model] = (
                None if wape is None else round(max(0.0, (1.0 - wape) * 100), 1)
            )

        scorable = {m: s for m, s in scores.items() if s is not None}
        if not scorable:
            return {
                "selected_model": None,
                "model_scores": scores,
                "justification": (
                    "Insufficient sales history to backtest any candidate; "
                    "falling back to moving-average baseline."
                ),
            }

        selected_model = max(scorable, key=scorable.get)
        return {
            "selected_model": selected_model,
            "model_scores": scores,
            "justification": ModelRouter.REASONS.get(
                demand_profile.get("demand_class", ""),
                "Optimal historical backtest accuracy.",
            ),
        }
