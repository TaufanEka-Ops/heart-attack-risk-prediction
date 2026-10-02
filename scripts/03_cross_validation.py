"""03 — Cross-validated model comparison.

The headline table.  Seven models, stratified 5-fold CV on all 4,238 rows,
reported as mean ± standard deviation across folds.  Accuracy, precision,
recall, F1, ROC-AUC, PR-AUC and the Brier score for each.

Writes: results/metrics_cv.csv, results/figures/08_model_comparison.png,
        results/transcripts/03_cross_validation.log
"""


from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))


import pandas as pd

from src.config import N_SPLITS, RESULTS_DIR
from src.data import load_dataset
from src.evaluate import METRIC_COLUMNS, cross_validate_models, majority_baseline
from src.models import get_models
from src.plots import model_comparison
from src.reporting import print_table, save_csv, transcript


def main() -> None:
    X, y = load_dataset()
    models = get_models()
    baseline = majority_baseline(y)

    print(f"protocol: StratifiedKFold(n_splits={N_SPLITS}, shuffle=True, random_state=42)")
    print(f"rows: {len(X)}   features: {X.shape[1]}   positives: {int(y.sum())} ({y.mean():.2%})")
    print(
        f"majority-class baseline: accuracy {baseline['accuracy_if_always_majority']:.4f}, "
        f"recall on positives {baseline['recall_on_positives']:.4f}, "
        f"PR-AUC (constant) {baseline['pr_auc_of_constant']:.4f}\n"
    )

    table = cross_validate_models(models, X, y, n_splits=N_SPLITS)

    # A reader-friendly view first, then the full precision table.
    pretty = pd.DataFrame(
        {
            "model": table["model"],
            "accuracy": table.apply(lambda r: f"{r['accuracy_mean']:.3f} ± {r['accuracy_std']:.3f}", axis=1),
            "recall": table.apply(lambda r: f"{r['recall_mean']:.3f} ± {r['recall_std']:.3f}", axis=1),
            "precision": table.apply(lambda r: f"{r['precision_mean']:.3f} ± {r['precision_std']:.3f}", axis=1),
            "f1": table.apply(lambda r: f"{r['f1_mean']:.3f} ± {r['f1_std']:.3f}", axis=1),
            "roc_auc": table.apply(lambda r: f"{r['roc_auc_mean']:.3f} ± {r['roc_auc_std']:.3f}", axis=1),
            "pr_auc": table.apply(lambda r: f"{r['pr_auc_mean']:.3f} ± {r['pr_auc_std']:.3f}", axis=1),
            "brier": table.apply(lambda r: f"{r['brier_mean']:.3f} ± {r['brier_std']:.3f}", axis=1),
        }
    )
    print_table(pretty, title=f"=== {N_SPLITS}-fold cross-validated performance (mean ± std) ===")

    ranked = table.sort_values("pr_auc_mean", ascending=False).reset_index(drop=True)
    print("=== ranking by PR-AUC (the metric that matters at 15% prevalence) ===")
    for rank, (_, row) in enumerate(ranked.iterrows(), start=1):
        print(
            f"  {rank}. {row['model']:<32} "
            f"PR-AUC {row['pr_auc_mean']:.4f} ± {row['pr_auc_std']:.4f}   "
            f"ROC-AUC {row['roc_auc_mean']:.4f}   "
            f"recall {row['recall_mean']:.4f}"
        )
    print()

    save_csv(table, RESULTS_DIR / "metrics_cv.csv")
    print(f"wrote {model_comparison(table, 'pr_auc')}")
    print(f"wrote {model_comparison(table, 'roc_auc', '08b_model_comparison_roc.png')}")

    best = ranked.iloc[0]
    print()
    print(f"best model by PR-AUC: {best['model']}")
    print(f"  it beats the no-skill PR-AUC of {baseline['pr_auc_of_constant']:.4f} "
          f"by {best['pr_auc_mean'] - baseline['pr_auc_of_constant']:.4f} absolute")


if __name__ == "__main__":
    with transcript("03_cross_validation"):
        main()
