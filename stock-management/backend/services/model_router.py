"""
model_router.py
WAPE-score-driven model selection with tie-break rule and dynamic explanations.
Spec: 02_pipeline_workflow_v3.md §2.3 / 03_addendum_spec_v3.md §3.4

Tie-break rule (when top-2 models are within TIE_THRESHOLD score points):
    Prefer the simpler model in this order:
    SBA(1) > TSB(2) > ETS(3) > SARIMAX(4) > LightGBM(5) > Chronos-2(6)

Explanation:
    Built dynamically from the selected model + actual score + demand class,
    not from a static lookup dict. Tie-break events are surfaced in the text.
"""


# Score difference within which tie-break applies (units: accuracy score 0-100)
TIE_THRESHOLD = 1.0


class ModelRouter:
    MODEL_REGISTRY = ["Chronos-2", "LightGBM", "SARIMAX", "ETS", "TSB", "SBA"]

    # Simplicity rank: lower number = simpler / more interpretable.
    # Used only for tie-breaking — never for primary selection.
    SIMPLICITY_RANK: dict[str, int] = {
        "SBA":      1,
        "TSB":      2,
        "ETS":      3,
        "SARIMAX":  4,
        "LightGBM": 5,
        "Chronos-2": 6,
    }

    # Short human-readable model descriptions for the explanation text
    MODEL_DESCRIPTIONS: dict[str, str] = {
        "LightGBM":  "LightGBM gradient-boosted regressor (lag features)",
        "SARIMAX":   "SARIMAX state-space model (weekly seasonality, AIC-selected orders)",
        "ETS":       "ETS Holt-Winters exponential smoothing",
        "TSB":       "TSB (Teunter-Syntetos-Babai) intermittent-demand model",
        "SBA":       "SBA (Syntetos-Boylan Approximation) intermittent-demand model",
        "Chronos-2": "Chronos-2 recency-weighted moving average",
    }

    # Demand-class context for explanation prose
    _DEMAND_CONTEXT: dict[str, str] = {
        "Smooth":       "continuous, stable demand",
        "Intermittent": "sporadic demand with frequent zero periods",
        "Erratic":      "high-frequency demand with volatile basket sizes",
        "Lumpy":        "sparse demand with highly variable order quantities",
        "Zero Demand":  "no recorded sales in the evaluation window",
    }

    @staticmethod
    def _build_explanation(
        selected_model: str,
        score: float,
        demand_class: str,
        runner_up: str | None,
        runner_up_score: float | None,
        was_tiebroken: bool,
    ) -> str:
        """
        Constructs a human-readable justification from actual selection result.
        Always based on the real winner and real score — never a static lookup.
        """
        desc = ModelRouter.MODEL_DESCRIPTIONS.get(selected_model, selected_model)
        ctx  = ModelRouter._DEMAND_CONTEXT.get(demand_class, demand_class)

        parts = [
            f"{desc} achieved the highest walk-forward backtest accuracy "
            f"(score {score:.1f}/100) on this SKU's {ctx}."
        ]

        if was_tiebroken and runner_up and runner_up_score is not None:
            runner_desc = ModelRouter.MODEL_DESCRIPTIONS.get(runner_up, runner_up)
            parts.append(
                f"Tie-break applied: {runner_desc} scored {runner_up_score:.1f}/100 "
                f"(within {TIE_THRESHOLD:.0f} pt of {selected_model}); "
                f"{selected_model} preferred as the simpler model."
            )

        return " ".join(parts)

    @staticmethod
    def route(demand_profile: dict, backtest_wape: dict) -> dict:
        """
        Converts WAPE values to scores, picks the best scorer with tie-break.

        Score_m = max(0, (1 − WAPE_m) × 100) if WAPE_m is defined, else null.

        Tie-break: when the top-2 scorable models are within TIE_THRESHOLD points,
        the simpler model (lower SIMPLICITY_RANK) wins.

        Parameters
        ----------
        demand_profile : dict from DemandClassifier.analyze() — uses demand_class
        backtest_wape  : {model_name: wape_float_or_None}

        Returns
        -------
        dict with selected_model, model_scores, justification
        """
        # Step 1: Convert WAPE → score (0-100, None if no data)
        scores: dict[str, float | None] = {}
        for model, wape in backtest_wape.items():
            scores[model] = (
                None if wape is None else round(max(0.0, (1.0 - wape) * 100), 1)
            )

        scorable = {m: s for m, s in scores.items() if s is not None}

        # Step 2: Insufficient data path
        if not scorable:
            return {
                "selected_model": None,
                "model_scores": scores,
                "justification": (
                    "Insufficient sales history to backtest any candidate model; "
                    "falling back to moving-average baseline."
                ),
            }

        # Step 3: Sort by score descending, simplicity rank ascending
        ranked = sorted(
            scorable.items(),
            key=lambda kv: (-kv[1], ModelRouter.SIMPLICITY_RANK.get(kv[0], 99)),
        )

        best_model, best_score = ranked[0]
        runner_up = ranked[1][0] if len(ranked) > 1 else None
        runner_up_score = ranked[1][1] if len(ranked) > 1 else None

        # Step 4: Check if tie-break was the deciding factor
        # (runner-up is within TIE_THRESHOLD AND it is simpler than best_model)
        was_tiebroken = False
        if (
            runner_up is not None
            and runner_up_score is not None
            and abs(best_score - runner_up_score) <= TIE_THRESHOLD
            and ModelRouter.SIMPLICITY_RANK.get(best_model, 99)
            < ModelRouter.SIMPLICITY_RANK.get(runner_up, 99)
        ):
            # Tie was broken in favour of best_model via simplicity — flag it
            was_tiebroken = True

        demand_class = demand_profile.get("demand_class", "")
        justification = ModelRouter._build_explanation(
            best_model, best_score, demand_class,
            runner_up, runner_up_score, was_tiebroken,
        )

        return {
            "selected_model": best_model,
            "model_scores": scores,
            "justification": justification,
        }
