"""
check_scenario_columns.py
Diagnostic: does the demo data have real price/promotion/discount variation
to learn from, or are these fields static per SKU?

This determines whether a "what-if price/promotion" feature can be built on
real empirical elasticity vs. must be a clearly-labeled assumed heuristic.

Run from stock-management/backend/:
    python check_scenario_columns.py
"""

import pandas as pd

CSV_PATH = "data/demo_inventory.csv"


def main():
    df = pd.read_csv(CSV_PATH)

    candidate_cols = [
        "Price", "Discount", "Holiday/Promotion",
        "Competitor Pricing", "Weather Condition",
    ]
    candidate_cols = [c for c in candidate_cols if c in df.columns]

    print(f"Checking variance for: {candidate_cols}\n")
    print(f"{'SKU':<10} " + " ".join(f"{c[:16]:<18}" for c in candidate_cols))
    print("-" * (10 + 19 * len(candidate_cols)))

    for sku, group in df.groupby("Product ID"):
        row = f"{sku:<10} "
        for col in candidate_cols:
            n_unique = group[col].nunique()
            if n_unique <= 1:
                summary = f"CONSTANT ({group[col].iloc[0]})"
            else:
                summary = f"{n_unique} values"
            row += f"{summary:<18} "
        print(row)

    print("\nInterpretation:")
    print("- 'CONSTANT' = no historical variation for this SKU -> cannot")
    print("  estimate empirical elasticity from this column, would need an")
    print("  assumed/heuristic value instead, clearly labeled as such.")
    print("- 'N values' = real variation exists -> a regression against")
    print("  Units Sold could extract a genuine, defensible relationship.")


if __name__ == "__main__":
    main()