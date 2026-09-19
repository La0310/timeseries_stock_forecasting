"""
test_classifier.py
Quick smoke test: run DemandClassifier.analyze() against demo_inventory.csv
for every SKU and print the classification results.

Run from stock-management/backend/:
    python test_classifier.py
"""

import pandas as pd
from services.demand_classifier import DemandClassifier

CSV_PATH = "data/demo_inventory.csv"


def main():
    df = pd.read_csv(CSV_PATH)

    print(f"Loaded {len(df)} rows, columns: {df.columns.tolist()}\n")

    sku_col = "Product ID"
    sales_col = "Units Sold"

    if sku_col is None or sales_col is None:
        print("Could not auto-detect columns. Please check manually:")
        print(df.head())
        print("\nSet sku_col / sales_col explicitly at the top of this script and rerun.")
        return

    print(f"Using sku_col='{sku_col}', sales_col='{sales_col}'\n")
    print(f"{'SKU':<10} {'Class':<14} {'ADI':<7} {'CV2':<7} {'ZeroRatio':<10} {'DailyStd':<10} {'Weekly':<20} {'Quarterly':<20}")
    print("-" * 110)

    for sku, group in df.groupby(sku_col):
        series = group[sales_col].astype(float)
        result = DemandClassifier.analyze(series)
        s = result["seasonality"]
        print(
            f"{sku:<10} {result['demand_class']:<14} {result['adi']:<7} "
            f"{result['cv_squared']:<7} {result['zero_demand_ratio']:<10} "
            f"{result['daily_sales_std']:<10} "
            f"{s['weekly_label']} ({s['weekly_acf']}){'':<5} "
            f"{s['quarterly_label']} ({s['quarterly_acf']})"
        )


if __name__ == "__main__":
    main()