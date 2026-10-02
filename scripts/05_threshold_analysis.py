"""05 — Decision-threshold analysis: from a probability to a referral decision.

A model that ranks well can still be useless at the default 0.5 cut-off.  The
original notebook never left 0.5, and never asked what the operating
characteristic would be at a threshold a clinic could actually use.

This script answers two questions on the *out-of-fold* probabilities:
  * which threshold maximises F1?
  * what is the least selective → most selective threshold that still catches
    70% of the people who go on to develop CHD, and how many people must be
    flagged to do it?

Writes: results/threshold_analysis.csv, results/thresholds_for_target_recall.csv,
        results/figures/09_thresholds.png,
        results/transcripts/05_threshold_analysis.log
"""


from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))


import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from src.config import RESULTS_DIR
from src.evaluate import threshold_table, thresholds_for_target_recall
from src.plots import threshold_curves
from src.reporting import print_table, save_csv, transcript

TARGET_RECALL = 0.70


def main() -> None:
    oof_path = RESULTS_DIR / "oof_probabilities.csv"
    cv_path = RESULTS_DIR / "metrics_cv.csv"
    if not oof_path.exists():
        raise SystemExit(f"{oof_path} missing — run scripts/04_holdout_evaluation.py first")

    oof = pd.read_csv(oof_path)
    y_true = oof["y_true"].to_numpy()

    if cv_path.exists():
        cv = pd.read_csv(cv_path)
        best = cv.sort_values("pr_auc_mean", ascending=False).iloc[0]["model"]
    else:
        candidates = [c for c in oof.columns if c != "y_true"]
        best = max(candidates, key=lambda c: average_precision_score(y_true, oof[c]))
    print(f"analysing: {best}  (out-of-fold probabilities, n={len(oof)})\n")

    proba = oof[best].to_numpy()

    table = threshold_table(y_true, proba)
    print_table(table, title="=== precision / recall by decision threshold ===")
    save_csv(table, RESULTS_DIR / "threshold_analysis.csv")

    default_row = table.loc[(table["threshold"] - 0.5).abs().idxmin()]
    best_f1_row = table.loc[table["f1"].idxmax()]
    print("=== the cost of the silent default ===")
    print(
        f"  threshold 0.50 : precision {default_row['precision']:.3f}, "
        f"recall {default_row['recall']:.3f}, F1 {default_row['f1']:.3f}, "
        f"flags {default_row['flagged_rate']:.1%} of everyone screened "
        f"({int(default_row['tp'])} true positives caught, {int(default_row['fn'])} missed)"
    )
    print(
        f"  threshold {best_f1_row['threshold']:.2f} : precision {best_f1_row['precision']:.3f}, "
        f"recall {best_f1_row['recall']:.3f}, F1 {best_f1_row['f1']:.3f}, "
        f"flags {best_f1_row['flagged_rate']:.1%} of everyone screened"
    )
    print()

    print(f"=== operating points that reach {TARGET_RECALL:.0%} sensitivity ===")
    recall_table = thresholds_for_target_recall(y_true, proba, TARGET_RECALL)
    if recall_table.empty:
        print(f"  no threshold in [0.01, 0.99] reaches recall {TARGET_RECALL:.0%}")
    else:
        print_table(recall_table)
        row = recall_table.iloc[0]
        flagged = row["tp"] + row["fp"]
        print(
            f"  To catch {TARGET_RECALL:.0%} of the {int(y_true.sum())} people who develop CHD\n"
            f"  you must flag {flagged} of {len(y_true)} ({flagged / len(y_true):.1%} of the cohort),\n"
            f"  and {int(row['fp'])} of those flags will be false alarms "
            f"(precision {row['precision']:.3f})."
        )
    save_csv(recall_table, RESULTS_DIR / "thresholds_for_target_recall.csv")

    print()
    print(f"wrote {threshold_curves(table)}")

    # Sensitivity at a few fixed operating points, useful in the README.
    print()
    print("=== summary line at fixed thresholds ===")
    for t in [0.5, float(best_f1_row["threshold"]), 0.2, 0.3]:
        row = table.loc[(table["threshold"] - t).abs().idxmin()]
        print(
            f"  tau={row['threshold']:.2f}  precision={row['precision']:.3f}  "
            f"recall={row['recall']:.3f}  f1={row['f1']:.3f}  flagged={row['flagged_rate']:.1%}"
        )


if __name__ == "__main__":
    with transcript("05_threshold_analysis"):
        main()
