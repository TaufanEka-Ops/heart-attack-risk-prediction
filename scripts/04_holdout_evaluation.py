"""04 — Held-out evaluation, out-of-fold probabilities, and diagnostics.

Reports a single stratified 80/20 split (matching the original notebook's
protocol) and, separately, the out-of-fold probabilities from 5-fold CV.  The
pooled out-of-fold vector is what the ROC, PR and calibration figures use: every
one of its 4,238 values comes from a model that never saw that row.

Writes: results/metrics_holdout.csv, results/oof_probabilities.csv,
        results/confusion_matrices.csv,
        results/figures/04_roc_curves.png, 05_pr_curves.png,
        06_confusion_matrix.png, 07_calibration.png,
        results/transcripts/04_holdout_evaluation.log
"""


from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))


import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

from src.config import N_SPLITS, RESULTS_DIR
from src.data import load_dataset
from src.evaluate import (
    METRIC_COLUMNS,
    calibration_summary,
    holdout_evaluate,
    out_of_fold_probabilities,
)
from src.models import get_models
from src.plots import calibration, confusion_matrix_plot, pr_curves, roc_curves
from src.reporting import print_table, save_csv, transcript


def main() -> None:
    X, y = load_dataset()
    models = get_models()

    # ------------------------------------------------------------------ #
    # 1. Single stratified 80/20 holdout, as in the original protocol      #
    # ------------------------------------------------------------------ #
    holdout, bundle = holdout_evaluate(models, X, y)
    print(f"holdout: {len(bundle['X_train'])} train / {len(bundle['X_test'])} test, stratified")
    print(
        f"positives: train {int(bundle['y_train'].sum())} "
        f"({bundle['y_train'].mean():.2%}), test {int(bundle['y_test'].sum())} "
        f"({bundle['y_test'].mean():.2%})\n"
    )
    print_table(holdout, title="=== held-out metrics (one stratified 80/20 split) ===")
    save_csv(holdout, RESULTS_DIR / "metrics_holdout.csv")

    # ------------------------------------------------------------------ #
    # 2. Out-of-fold probabilities from 5-fold CV                          #
    # ------------------------------------------------------------------ #
    print(f"generating out-of-fold probabilities ({N_SPLITS} folds x {len(models)} models)...")
    oof = pd.DataFrame({"y_true": np.asarray(y)})
    for name, pipeline in models.items():
        oof[name] = out_of_fold_probabilities(pipeline, X, y, n_splits=N_SPLITS)
    save_csv(oof, RESULTS_DIR / "oof_probabilities.csv")

    probabilities = {name: oof[name].to_numpy() for name in models}
    y_arr = oof["y_true"].to_numpy()

    # ------------------------------------------------------------------ #
    # 3. Calibration: are the probabilities on the risk scale?             #
    # ------------------------------------------------------------------ #
    calib = calibration_summary(
        y_arr, probabilities, reference_model="Logistic Regression"
    )
    print_table(
        calib,
        title="=== calibration (mean_predicted should match the observed rate) ===",
    )
    print(
        "Reading this table: 'Logistic Regression' predicts a mean risk of ~0.152\n"
        "against an observed 0.152 — it is calibrated in the large. The balanced\n"
        "variant predicts ~0.44 for the same people: identical ranking (see the\n"
        "spearman column), probabilities inflated ~3x. The extra positives it\n"
        "catches at the 0.5 cut-off come from that inflation, not from better\n"
        "discrimination — and the same effect is available for free by lowering\n"
        "the threshold on the calibrated model (scripts/05).\n"
    )
    save_csv(calib, RESULTS_DIR / "calibration_summary.csv")

    # ------------------------------------------------------------------ #
    # 4. Confusion matrices on the holdout, for every model                #
    # ------------------------------------------------------------------ #
    cm_rows = []
    for name, y_pred in bundle["predictions"].items():
        tn, fp, fn, tp = confusion_matrix(bundle["y_test"], y_pred, labels=[0, 1]).ravel()
        cm_rows.append(
            {
                "model": name,
                "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
                "accuracy": (tp + tn) / (tp + tn + fp + fn),
                "precision": precision_score(bundle["y_test"], y_pred, zero_division=0),
                "recall": recall_score(bundle["y_test"], y_pred, zero_division=0),
                "f1": f1_score(bundle["y_test"], y_pred, zero_division=0),
            }
        )
    cm_table = pd.DataFrame(cm_rows)
    print_table(cm_table, title="=== confusion matrix counts at threshold 0.5 (holdout) ===")
    save_csv(cm_table, RESULTS_DIR / "confusion_matrices.csv")

    # ------------------------------------------------------------------ #
    # 5. Figures                                                          #
    # ------------------------------------------------------------------ #
    best_name = holdout.sort_values("pr_auc", ascending=False).iloc[0]["model"]
    print(f"best model on the holdout by PR-AUC: {best_name}\n")

    print(f"wrote {roc_curves(y_arr, probabilities)}")
    print(f"wrote {pr_curves(y_arr, probabilities)}")
    print(
        "wrote "
        + confusion_matrix_plot(
            bundle["y_test"],
            bundle["predictions"][best_name],
            title=f"Confusion matrix — {best_name} (holdout, threshold 0.5)",
        )
    )
    print(f"wrote {calibration(y_arr, probabilities)}")


if __name__ == "__main__":
    with transcript("04_holdout_evaluation"):
        main()
