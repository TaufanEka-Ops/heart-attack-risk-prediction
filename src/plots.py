"""Every figure in ``results/figures``.

Matplotlib is put on the ``Agg`` backend here so the scripts run headless on a
server.  Figures are written to disk, never shown.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # noqa: E402  (must precede pyplot import)

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.calibration import calibration_curve  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)

from .config import FIGURES_DIR  # noqa: E402

__all__ = [
    "class_balance",
    "missingness",
    "correlation_heatmap",
    "roc_curves",
    "pr_curves",
    "confusion_matrix_plot",
    "calibration",
    "model_comparison",
    "threshold_curves",
]

PALETTE = [
    "#4C72B0", "#DD8452", "#55A868", "#C44E52",
    "#8172B3", "#937860", "#DA8BC3", "#8C8C8C",
]


def _save(fig, name: str) -> str:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / name
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def class_balance(y, name: str = "01_class_balance.png") -> str:
    fig, ax = plt.subplots(figsize=(5, 4))
    counts = y.value_counts().sort_index()
    bars = ax.bar(["No CHD (0)", "CHD (1)"], counts.values, color=["#4C72B0", "#C44E52"])
    for bar, count in zip(bars, counts.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            count,
            f"{count:,}\n{count / len(y):.1%}",
            ha="center",
            va="bottom",
            fontsize=10,
        )
    ax.set_ylabel("Participants")
    ax.set_title("Ten-year CHD class balance")
    ax.set_ylim(0, counts.max() * 1.18)
    ax.grid(axis="y", alpha=0.3)
    return _save(fig, name)


def missingness(table: pd.DataFrame, name: str = "02_missingness.png") -> str:
    fig, ax = plt.subplots(figsize=(6, 4))
    table = table.sort_values("missing")
    bars = ax.barh(table.index, table["pct_missing"], color="#DD8452")
    total = int(table["missing"].sum())
    for bar, (col, row) in zip(bars, table.iterrows()):
        ax.text(
            bar.get_width() + 0.05,
            bar.get_y() + bar.get_height() / 2,
            f"{int(row['missing'])} ({row['pct_missing']:.2f}%)",
            va="center",
            fontsize=9,
        )
    ax.set_xlabel("% missing")
    ax.set_title(f"Missing values by column (n={total} cells)")
    ax.set_xlim(0, table["pct_missing"].max() * 1.45)
    ax.grid(axis="x", alpha=0.3)
    return _save(fig, name)


def correlation_heatmap(X: pd.DataFrame, y: pd.Series, name: str = "03_correlation.png") -> str:
    frame = X.copy()
    frame["TenYearCHD"] = np.asarray(y)
    corr = frame.corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(9, 7.5))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr)))
    ax.set_yticks(range(len(corr)))
    ax.set_xticklabels(corr.columns, rotation=90, fontsize=8)
    ax.set_yticklabels(corr.columns, fontsize=8)
    for i in range(len(corr)):
        for j in range(len(corr)):
            value = corr.values[i, j]
            ax.text(
                j, i, f"{value:.2f}", ha="center", va="center",
                fontsize=6, color="white" if abs(value) > 0.5 else "black",
            )
    fig.colorbar(im, ax=ax, shrink=0.8, label="Pearson r")
    ax.set_title("Feature correlation (last row/column = target)")
    return _save(fig, name)


def roc_curves(y_true, probabilities: dict, name: str = "04_roc_curves.png") -> str:
    fig, ax = plt.subplots(figsize=(6.5, 6))
    for (label, proba), color in zip(probabilities.items(), PALETTE):
        if proba is None:
            continue
        fpr, tpr, _ = roc_curve(y_true, proba)
        from sklearn.metrics import auc

        ax.plot(fpr, tpr, color=color, lw=1.8, label=f"{label} (AUC={auc(fpr, tpr):.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Chance")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curves — out-of-fold probabilities")
    ax.legend(fontsize=7.5, loc="lower right")
    ax.grid(alpha=0.3)
    return _save(fig, name)


def pr_curves(y_true, probabilities: dict, name: str = "05_pr_curves.png") -> str:
    base = float(np.mean(y_true))
    fig, ax = plt.subplots(figsize=(6.5, 6))
    from sklearn.metrics import average_precision_score

    for (label, proba), color in zip(probabilities.items(), PALETTE):
        if proba is None:
            continue
        precision, recall, _ = precision_recall_curve(y_true, proba)
        ap = average_precision_score(y_true, proba)
        ax.plot(recall, precision, color=color, lw=1.8, label=f"{label} (AP={ap:.3f})")
    ax.axhline(base, color="k", ls="--", lw=1, label=f"No-skill baseline ({base:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision–recall curves — out-of-fold probabilities")
    ax.legend(fontsize=7.5)
    ax.grid(alpha=0.3)
    return _save(fig, name)


def confusion_matrix_plot(
    y_true, y_pred, labels=("No CHD", "CHD"), title="Confusion matrix", name="06_confusion_matrix.png"
) -> str:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], labels)
    ax.set_yticks([0, 1], labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    for i in range(2):
        for j in range(2):
            ax.text(
                j, i, f"{cm[i, j]:,}", ha="center", va="center", fontsize=13,
                color="white" if cm[i, j] > cm.max() / 2 else "black",
            )
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title(title)
    return _save(fig, name)


def calibration(y_true, probabilities: dict, n_bins: int = 10, name: str = "07_calibration.png") -> str:
    fig, ax = plt.subplots(figsize=(6, 5.5))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfectly calibrated")
    for (label, proba), color in zip(probabilities.items(), PALETTE):
        if proba is None:
            continue
        frac_pos, mean_pred = calibration_curve(y_true, proba, n_bins=n_bins, strategy="quantile")
        ax.plot(mean_pred, frac_pos, marker="o", ms=4, color=color, lw=1.5, label=label)
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed fraction of positives")
    ax.set_title("Calibration curves (quantile bins) — out-of-fold")
    ax.legend(fontsize=7.5)
    ax.grid(alpha=0.3)
    return _save(fig, name)


def model_comparison(cv_table: pd.DataFrame, metric: str = "pr_auc", name: str = "08_model_comparison.png") -> str:
    ordered = cv_table.sort_values(f"{metric}_mean")
    fig, ax = plt.subplots(figsize=(7.5, 0.6 * len(ordered) + 2))
    ax.barh(
        ordered["model"],
        ordered[f"{metric}_mean"],
        xerr=ordered[f"{metric}_std"],
        color="#4C72B0",
        error_kw={"ecolor": "#333333", "capsize": 3},
    )
    for y_pos, (mean, std) in enumerate(zip(ordered[f"{metric}_mean"], ordered[f"{metric}_std"])):
        ax.text(mean + std + 0.005, y_pos, f"{mean:.3f} ± {std:.3f}", va="center", fontsize=8)
    ax.set_xlabel(f"{metric.upper().replace('_', '-')} (5-fold CV mean ± std)")
    ax.set_title("Model comparison")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    return _save(fig, name)


def threshold_curves(table: pd.DataFrame, name: str = "09_thresholds.png") -> str:
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(table["threshold"], table["precision"], label="Precision", color="#4C72B0")
    ax.plot(table["threshold"], table["recall"], label="Recall (sensitivity)", color="#C44E52")
    ax.plot(table["threshold"], table["f1"], label="F1", color="#55A868")
    ax.plot(table["threshold"], table["flagged_rate"], label="Fraction flagged", color="#8172B3", ls=":")
    ax.axvline(0.5, color="k", ls="--", lw=1, label="Default threshold 0.5")
    best = table.loc[table["f1"].idxmax()]
    ax.axvline(best["threshold"], color="#DD8452", ls="-.", lw=1.2,
               label=f"Best F1 @ {best['threshold']:.2f}")
    ax.set_xlabel("Decision threshold")
    ax.set_ylabel("Score")
    ax.set_title("Threshold trade-offs — out-of-fold probabilities")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    return _save(fig, name)
