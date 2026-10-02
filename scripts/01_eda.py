"""01 — Exploratory data analysis.

Answers three questions before any modelling:
  1. What does the data contract look like? (rows, positives, missing cells)
  2. Where are the missing values, and what kind are they?
  3. Which features move together, and which correlate with the target?

Writes: results/figures/01_class_balance.png, 02_missingness.png,
        03_correlation.png, results/missingness.csv, results/transcripts/01_eda.log
"""


from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))


import pandas as pd

from src.config import RESULTS_DIR
from src.data import data_contract, load_dataset, load_raw, missingness_table
from src.evaluate import majority_baseline
from src.plots import class_balance, correlation_heatmap, missingness
from src.reporting import print_table, save_csv, transcript


def main() -> None:
    df = load_raw()
    X, y = load_dataset()

    # ------------------------------------------------------------------ #
    # 1. Data contract                                                    #
    # ------------------------------------------------------------------ #
    contract = data_contract(df)
    print("=== data contract ===")
    for key, value in contract.items():
        print(f"  {key:<26} {value}")

    baseline = majority_baseline(y)
    print()
    print("=== what accuracy alone buys you ===")
    print(f"  positive rate                     {baseline['positive_rate']:.4f}")
    print(f"  accuracy of always predicting 0   {baseline['accuracy_if_always_majority']:.4f}")
    print(f"  recall of always predicting 0     {baseline['recall_on_positives']:.4f}")
    print()

    # ------------------------------------------------------------------ #
    # 2. Missingness                                                      #
    # ------------------------------------------------------------------ #
    table = missingness_table(df)
    print_table(table.reset_index(names="column"), title="=== missing values (n=4238) ===")
    print(f"total missing cells: {int(table['missing'].sum())} "
          f"({table['missing'].sum() / df.size:.2%} of the matrix)")
    print(f"rows that are complete: {int((~df.isna().any(axis=1)).sum())}")
    print()

    # Missingness is concentrated in glucose; check whether it is random with
    # respect to the target, which decides how much we should worry.
    print("=== is missingness related to the outcome? ===")
    for col in table.index:
        is_missing = df[col].isna()
        rate_missing = df.loc[is_missing, "TenYearCHD"].mean()
        rate_present = df.loc[~is_missing, "TenYearCHD"].mean()
        print(
            f"  {col:<12} CHD rate when missing {rate_missing:.3f} "
            f"vs present {rate_present:.3f}  (n_missing={int(is_missing.sum())})"
        )
    print()

    # ------------------------------------------------------------------ #
    # 3. Correlations                                                     #
    # ------------------------------------------------------------------ #
    corr_with_target = (
        pd.concat([X, y], axis=1)
        .corr(numeric_only=True)["TenYearCHD"]
        .drop("TenYearCHD")
        .sort_values(key=abs, ascending=False)
    )
    print_table(
        corr_with_target.rename_axis("feature").reset_index(name="corr_with_target"),
        title="=== features ranked by |correlation| with TenYearCHD ===",
    )

    save_csv(table.reset_index(names="column"), RESULTS_DIR / "missingness.csv")
    # ------------------------------------------------------------------ #
    # 4. Figures                                                          #
    # ------------------------------------------------------------------ #
    print()
    for path in (
        class_balance(y),
        missingness(table),
        correlation_heatmap(X, y),
    ):
        print(f"wrote figure {path}")


if __name__ == "__main__":
    with transcript("01_eda"):
        main()
