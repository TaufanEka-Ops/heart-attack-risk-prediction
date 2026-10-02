"""07 — Build and execute the portfolio notebook.

The notebook is *generated* from this script rather than hand-edited, so the
cells and their wording live under version control in a diffable form, and then
executed with ``nbclient`` so the committed ``.ipynb`` carries real outputs.

Running this rewrites ``notebooks/01_eda_and_cv.ipynb`` in place.

Writes: notebooks/01_eda_and_cv.ipynb, results/transcripts/07_notebook.log
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import nbformat
from nbclient import NotebookClient

from src.config import PROJECT_ROOT
from src.reporting import transcript

NOTEBOOK_PATH = PROJECT_ROOT / "notebooks" / "01_eda_and_cv.ipynb"

BOOTSTRAP = """\
import pathlib, sys
ROOT = pathlib.Path.cwd().parent          # notebooks/ -> repository root
sys.path.insert(0, str(ROOT))
print("repository root:", ROOT)"""

CELLS: list[tuple[str, str]] = [
    ("markdown", """\
# Framingham CHD risk — data, imbalance, and the threshold nobody chose

A short, executed walkthrough of the three things that decide what this model is
worth:

1. **The base rate.** 15% of participants develop CHD, so accuracy is close to
   useless as a headline.
2. **Calibration.** `class_weight="balanced"` triples the predicted probabilities
   without changing the ranking at all.
3. **The decision threshold.** The 0.5 default catches almost nobody.

Every number below is computed here, from the raw CSV, with the same pipeline the
scripts use. Nothing is quoted from memory."""),
    ("code", BOOTSTRAP),
    ("code", """\
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve

from src.data import load_dataset, missingness_table
from src.evaluate import (
    calibration_summary,
    cross_validate_models,
    majority_baseline,
    threshold_table,
)
from src.models import get_models

pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 40)
plt.rcParams["figure.dpi"] = 110
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.3

X, y = load_dataset()
print(f"loaded {len(X):,} participants x {X.shape[1]} features")
print(f"positive class: {int(y.sum())} ({y.mean():.2%})")"""),
    ("markdown", """\
## 1. The data, and where it is missing

The loader returns the frame **with missing values intact**. Imputation happens
inside the model pipeline, on training folds only."""),
    ("code", """\
missingness_table(X).style.format({"pct_missing": "{:.2f}%"}).set_caption(
    "missing values by column"
)"""),
    ("markdown", """\
## 2. The number that decides everything

Before looking at any model, look at what a model that learns nothing scores."""),
    ("code", """\
baseline = majority_baseline(y)
print(f"positive rate                        {baseline['positive_rate']:.4f}")
print(f"accuracy of always predicting 'no'   {baseline['accuracy_if_always_majority']:.4f}")
print(f"recall of always predicting 'no'     {baseline['recall_on_positives']:.4f}")
print(f"PR-AUC of a constant predictor       {baseline['pr_auc_of_constant']:.4f}")

fig, ax = plt.subplots(figsize=(4.6, 3.6))
counts = y.value_counts().sort_index()
bars = ax.bar(["No CHD", "CHD"], counts.values, color=["#4C72B0", "#C44E52"])
for bar, c in zip(bars, counts.values):
    ax.text(bar.get_x() + bar.get_width() / 2, c, f"{c:,}\\n{c / len(y):.1%}",
            ha="center", va="bottom", fontsize=9)
ax.set_ylim(0, counts.max() * 1.2)
ax.set_ylabel("participants")
ax.set_title("A majority-class predictor scores 0.848 accuracy")
plt.show()"""),
    ("markdown", """\
## 3. Cross-validated comparison

Three models are enough to make the point: the majority baseline, the original
notebook's logistic regression, and the same model with `class_weight="balanced"`.
Preprocessing (median / most-frequent imputation, then scaling) is re-fitted
inside every fold."""),
    ("code", """\
all_models = get_models()
models = {k: all_models[k] for k in [
    "Dummy (majority class)",
    "Logistic Regression",
    "Logistic Regression (balanced)",
]}

cv = cross_validate_models(models, X, y, n_splits=5)
display = pd.DataFrame({
    "model": cv["model"],
    "accuracy": cv.apply(lambda r: f"{r.accuracy_mean:.4f} ± {r.accuracy_std:.4f}", axis=1),
    "precision": cv.apply(lambda r: f"{r.precision_mean:.4f} ± {r.precision_std:.4f}", axis=1),
    "recall": cv.apply(lambda r: f"{r.recall_mean:.4f} ± {r.recall_std:.4f}", axis=1),
    "f1": cv.apply(lambda r: f"{r.f1_mean:.4f} ± {r.f1_std:.4f}", axis=1),
    "roc_auc": cv.apply(lambda r: f"{r.roc_auc_mean:.4f} ± {r.roc_auc_std:.4f}", axis=1),
    "pr_auc": cv.apply(lambda r: f"{r.pr_auc_mean:.4f} ± {r.pr_auc_std:.4f}", axis=1),
})
display"""),
    ("markdown", """\
**Read the `recall` column.** The logistic regression has the best accuracy in the
table (0.8551) and catches **8%** of the people who go on to develop CHD. The
balanced variant catches 67% and its accuracy drops to 0.67 — but note that its
ROC-AUC is *identical*. Something other than discrimination is changing.

Below, out-of-fold probabilities: every participant scored by a model that never
saw them."""),
    ("code", """\
from src.evaluate import out_of_fold_probabilities

oof = pd.DataFrame({"y_true": np.asarray(y)})
for name, pipe in models.items():
    oof[name] = out_of_fold_probabilities(pipe, X, y, n_splits=5)

calib = calibration_summary(
    oof["y_true"].to_numpy(),
    {c: oof[c].to_numpy() for c in models},
    reference_model="Logistic Regression",
)
calib.style.format({
    "observed_rate": "{:.4f}", "mean_predicted": "{:.4f}",
    "calibration_ratio": "{:.3f}", "brier": "{:.4f}",
    "ece_quantile_10": "{:.4f}", "spearman_vs_reference": "{:.4f}",
}).set_caption("calibration, out-of-fold")"""),
    ("code", """\
fig, ax = plt.subplots(figsize=(5.2, 4.6))
ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfectly calibrated")
for name, color in [("Logistic Regression", "#4C72B0"),
                    ("Logistic Regression (balanced)", "#C44E52")]:
    frac, mean_pred = calibration_curve(oof["y_true"], oof[name], n_bins=10, strategy="quantile")
    ax.plot(mean_pred, frac, "o-", color=color, ms=4, lw=1.5, label=name)
ax.axvline(oof["y_true"].mean(), color="#555555", ls=":", lw=1, label="observed rate 0.152")
ax.set_xlabel("mean predicted probability")
ax.set_ylabel("observed fraction with CHD")
ax.set_title("Balancing inflates the probabilities, not the ranking")
ax.legend(fontsize=7.5, loc="upper left")
plt.show()"""),
    ("markdown", """\
The Spearman column above is ~0.998: the balanced model puts participants in
almost exactly the same order. It multiplies every probability by ~2.9. That
inflation is the *entire* reason it catches more cases at a fixed 0.5 cut-off —
and the same effect is available for free, and reversibly, by moving the
threshold.

## 4. The threshold nobody chose

Using the well-calibrated model's out-of-fold probabilities."""),
    ("code", """\
proba = oof["Logistic Regression"].to_numpy()
y_true = oof["y_true"].to_numpy()
table = threshold_table(y_true, proba)
table.style.format({
    "threshold": "{:.2f}", "precision": "{:.3f}", "recall": "{:.3f}",
    "f1": "{:.3f}", "flagged_rate": "{:.1%}",
}).set_caption("precision / recall by decision threshold")"""),
    ("code", """\
fig, ax = plt.subplots(figsize=(6.4, 4.2))
ax.plot(table.threshold, table.precision, label="precision", color="#4C72B0")
ax.plot(table.threshold, table.recall, label="recall (sensitivity)", color="#C44E52")
ax.plot(table.threshold, table.f1, label="F1", color="#55A868")
ax.axvline(0.5, color="k", ls="--", lw=1, label="default 0.5")
ax.axvline(0.14, color="#DD8452", ls="-.", lw=1.2, label="0.14 (70% sensitivity)")
ax.set_xlabel("decision threshold")
ax.set_ylabel("score")
ax.set_title("The default threshold catches 8% of events")
ax.legend(fontsize=8)
plt.show()"""),
    ("code", """\
default = table.loc[(table.threshold - 0.5).abs().idxmin()]
sens70 = table.loc[(table.threshold - 0.14).abs().idxmin()]
print(f"threshold 0.50: recall {default.recall:.3f}, flags {default.flagged_rate:.1%} of everyone,"
      f" catches {int(default.tp)} of {int(y_true.sum())} events")
print(f"threshold 0.14: recall {sens70.recall:.3f}, flags {sens70.flagged_rate:.1%} of everyone,"
      f" catches {int(sens70.tp)} of {int(y_true.sum())} events,"
      f" {int(sens70.fp)} of those flags are false alarms")"""),
    ("markdown", """\
## What this notebook concludes

- **Accuracy is the wrong headline here.** A predictor that learns nothing scores
  0.848; the best model scores 0.8551 while missing 92% of events at the default
  threshold.
- **Discrimination is modest.** ROC-AUC ≈ 0.725, PR-AUC ≈ 0.347 against a
  no-skill 0.152.
- **Imbalance weighting is a relabelling.** `class_weight="balanced"` leaves the
  ranking unchanged (Spearman 0.998) and triples the predicted probabilities,
  costing a doubling of the Brier score (0.117 → 0.210).
- **The threshold is the real decision.** Reaching 70% sensitivity means flagging
  about 42% of the cohort, three quarters of them wrongly.

This is a risk-ranking tool, not a referral rule. The full analysis, the leakage
audit of the original notebook, and the tests are in the repository."""),
]


def build_notebook() -> nbformat.NotebookNode:
    notebook = nbformat.v4.new_notebook()
    notebook.cells = []
    for kind, source in CELLS:
        if kind == "markdown":
            notebook.cells.append(nbformat.v4.new_markdown_cell(source))
        else:
            notebook.cells.append(nbformat.v4.new_code_cell(source))
    notebook.metadata.update(
        {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python"},
        }
    )
    return notebook


def main() -> None:
    notebook = build_notebook()
    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)

    print(f"executing {len(notebook.cells)} cells ...")
    client = NotebookClient(
        notebook,
        timeout=900,
        kernel_name="python3",
        resources={"metadata": {"path": str(NOTEBOOK_PATH.parent)}},
        allow_errors=False,
    )
    client.execute()

    nbformat.write(notebook, NOTEBOOK_PATH)

    code_cells = [c for c in notebook.cells if c.cell_type == "code"]
    with_output = [c for c in code_cells if c.get("outputs")]
    print(f"wrote {NOTEBOOK_PATH.relative_to(PROJECT_ROOT)}")
    print(f"  {len(notebook.cells)} cells, {len(code_cells)} code cells, "
          f"{len(with_output)} carrying outputs")


if __name__ == "__main__":
    with transcript("07_notebook"):
        main()
