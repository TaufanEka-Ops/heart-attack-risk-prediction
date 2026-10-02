"""Evaluation protocol.

Two evaluations, on purpose:

* ``cross_validate_models`` — 5-fold stratified CV on the whole dataset.  This
  is where the headline numbers come from, because a single 80/20 split on 635
  positives is noisy.
* ``holdout_evaluate`` — one stratified 80/20 split, mirroring the original
  notebook's protocol, so the before/after comparison is like-for-like.

Metrics reported: accuracy, precision, recall, F1, ROC-AUC, PR-AUC (average
precision), and the Brier score.  Accuracy is reported but treated as the least
interesting column — see the module docstring of ``docs/methodology.md``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split

from .config import N_SPLITS, RANDOM_STATE, TARGET, TEST_SIZE

__all__ = [
    "METRIC_COLUMNS",
    "calculate_metrics",
    "calibration_summary",
    "expected_calibration_error",
    "make_split",
    "make_cv",
    "cross_validate_models",
    "holdout_evaluate",
    "out_of_fold_probabilities",
    "majority_baseline",
    "threshold_table",
]

METRIC_COLUMNS = [
    "accuracy",
    "precision",
    "recall",
    "f1",
    "roc_auc",
    "pr_auc",
    "brier",
]


def make_cv(n_splits: int = N_SPLITS) -> StratifiedKFold:
    """Stratified k-fold with a fixed seed.

    Stratification keeps the ~15% positive rate in every fold; without it a
    fold can end up with far fewer positives than the others and the fold
    variance becomes meaningless.
    """
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)


def make_split(X, y, test_size: float = TEST_SIZE):
    """Stratified 80/20 holdout split."""
    return train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=RANDOM_STATE
    )


def calculate_metrics(y_true, y_pred, y_proba=None) -> dict:
    """All metrics for one set of predictions.

    ``y_proba`` is the probability of the positive class.  Rank-based metrics
    (ROC-AUC, PR-AUC) and the Brier score are only computed when it is given.
    """
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }
    if y_proba is not None:
        metrics["roc_auc"] = roc_auc_score(y_true, y_proba)
        metrics["pr_auc"] = average_precision_score(y_true, y_proba)
        metrics["brier"] = brier_score_loss(y_true, y_proba)
    else:
        metrics["roc_auc"] = np.nan
        metrics["pr_auc"] = np.nan
        metrics["brier"] = np.nan
    return metrics


def cross_validate_models(models: dict, X, y, n_splits: int = N_SPLITS) -> pd.DataFrame:
    """Stratified CV for every model; returns mean and std per metric.

    Uses an explicit fold loop rather than ``sklearn.cross_validate`` so the
    per-fold numbers are available for the standard deviations, and so the
    fitted pipeline of each fold is available if a caller wants it.
    """
    cv = make_cv(n_splits)
    classes = np.unique(y)
    if len(classes) != 2:
        raise ValueError(f"expected a binary target, got classes {classes}")

    rows: list[dict] = []
    for name, pipeline in models.items():
        fold_metrics: list[dict] = []
        for train_idx, test_idx in cv.split(X, y):
            X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
            y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]

            fitted = clone(pipeline).fit(X_tr, y_tr)
            y_pred = fitted.predict(X_te)
            if hasattr(fitted, "predict_proba"):
                y_proba = fitted.predict_proba(X_te)[:, 1]
            else:
                y_proba = None
            fold_metrics.append(calculate_metrics(y_te, y_pred, y_proba))

        fold_df = pd.DataFrame(fold_metrics)
        record = {"model": name, "n_folds": n_splits}
        for metric in METRIC_COLUMNS:
            record[f"{metric}_mean"] = fold_df[metric].mean()
            record[f"{metric}_std"] = fold_df[metric].std(ddof=1)
        rows.append(record)

    return pd.DataFrame(rows)


def holdout_evaluate(models: dict, X, y, test_size: float = TEST_SIZE) -> tuple[pd.DataFrame, dict]:
    """Fit on the training split, score on the held-out split.

    Returns ``(metrics_df, bundle)`` where ``bundle`` holds the split and the
    fitted pipelines for downstream plotting.
    """
    X_train, X_test, y_train, y_test = make_split(X, y, test_size)
    rows, fitted_models, probabilities = [], {}, {}

    for name, pipeline in models.items():
        fitted = clone(pipeline).fit(X_train, y_train)
        y_pred = fitted.predict(X_test)
        y_proba = (
            fitted.predict_proba(X_test)[:, 1]
            if hasattr(fitted, "predict_proba")
            else None
        )
        fitted_models[name] = fitted
        probabilities[name] = y_proba
        row = calculate_metrics(y_test, y_pred, y_proba)
        row["model"] = name
        rows.append(row)

    metrics_df = pd.DataFrame(rows)[["model"] + METRIC_COLUMNS]
    bundle = {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "fitted": fitted_models,
        "probabilities": probabilities,
        "predictions": {n: m.predict(X_test) for n, m in fitted_models.items()},
    }
    return metrics_df, bundle


def out_of_fold_probabilities(pipeline, X, y, n_splits: int = N_SPLITS) -> np.ndarray:
    """Cross-validated predicted probabilities — one per row, never in-sample.

    Used for threshold selection, calibration curves and pooled ROC/PR curves.
    Every row is scored by a model that did not see it during fitting, so it is
    legitimate to choose a decision threshold from these numbers.
    """
    cv = make_cv(n_splits)
    return cross_val_predict(pipeline, X, y, cv=cv, method="predict_proba")[:, 1]


def _quantile_bins(proba: np.ndarray, n_bins: int) -> np.ndarray:
    """Bin index per observation using quantile edges (equal-count bins)."""
    edges = np.quantile(proba, np.linspace(0.0, 1.0, n_bins + 1))
    interior = np.unique(edges[1:-1])
    if interior.size == 0:
        return np.zeros_like(proba, dtype=int)
    return np.digitize(proba, interior)


def expected_calibration_error(y_true, y_proba, n_bins: int = 10) -> float:
    """Mean |observed rate − predicted rate| weighted by bin population.

    Quantile bins, so every bin holds the same number of people and a handful of
    extreme predictions cannot dominate the statistic.
    """
    y_true = np.asarray(y_true, dtype=float)
    proba = np.asarray(y_proba, dtype=float)
    bins = _quantile_bins(proba, n_bins)
    total = len(proba)
    ece = 0.0
    for b in np.unique(bins):
        mask = bins == b
        n = int(mask.sum())
        if n == 0:
            continue
        ece += (n / total) * abs(y_true[mask].mean() - proba[mask].mean())
    return float(ece)


def calibration_summary(
    y_true, probabilities: dict, n_bins: int = 10, reference_model: str | None = None
) -> pd.DataFrame:
    """Calibration-in-the-large, Brier score and ECE for each model.

    ``mean_predicted`` far from ``observed_rate`` means the model's probabilities
    are not on the risk scale, whatever their ranking quality.  This is the table
    that exposes ``class_weight="balanced"`` for what it is: a monotone rescaling
    of an otherwise unchanged ranking.
    """
    y_true = np.asarray(y_true)
    observed = float(y_true.mean())
    rows = []
    for name, proba in probabilities.items():
        if proba is None:
            continue
        proba = np.asarray(proba, dtype=float)
        ranks = pd.Series(proba).rank().to_numpy()
        rows.append(
            {
                "model": name,
                "observed_rate": observed,
                "mean_predicted": float(proba.mean()),
                "calibration_ratio": float(proba.mean() / observed) if observed else np.nan,
                "brier": brier_score_loss(y_true, proba),
                "ece_quantile_10": expected_calibration_error(y_true, proba, n_bins),
                "_ranks": ranks,
            }
        )
    table = pd.DataFrame(rows)

    # Pearson correlation of ranks == Spearman.  Measured against a reference
    # ranking (the plain, well-calibrated logistic regression by default):
    # same ranking with different probabilities means the "improvement" is
    # threshold plumbing, not discrimination.
    reference = None
    if reference_model and reference_model in set(table["model"]):
        reference = table.loc[table["model"] == reference_model, "_ranks"].iloc[0]
    if reference is None:
        for _, row in table.iterrows():
            if len(np.unique(row["_ranks"])) > 1:
                reference = row["_ranks"]
                break
    if reference is not None:
        spearman = []
        for ranks in table["_ranks"]:
            if len(np.unique(ranks)) < 2:
                spearman.append(np.nan)  # a constant score has no ranking
            else:
                spearman.append(float(np.corrcoef(reference, ranks)[0, 1]))
        table["spearman_vs_reference"] = spearman
    return table.drop(columns=["_ranks"])


def majority_baseline(y) -> dict:
    """The number accuracy alone cannot beat: always predicting the majority."""
    rate = float(np.mean(y))
    majority = int(rate >= 0.5)
    return {
        "majority_class": majority,
        "positive_rate": rate,
        "accuracy_if_always_majority": max(rate, 1 - rate),
        "recall_on_positives": 1.0 if majority == 1 else 0.0,
        "pr_auc_of_constant": rate,
    }


def threshold_table(y_true, y_proba, thresholds=None) -> pd.DataFrame:
    """Precision / recall / F1 at a range of decision thresholds."""
    if thresholds is None:
        thresholds = np.round(np.arange(0.05, 0.96, 0.05), 2)
    rows = []
    for t in thresholds:
        y_pred = (np.asarray(y_proba) >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        rows.append(
            {
                "threshold": float(t),
                "precision": precision_score(y_true, y_pred, zero_division=0),
                "recall": recall_score(y_true, y_pred, zero_division=0),
                "f1": f1_score(y_true, y_pred, zero_division=0),
                "flagged_rate": float(np.mean(y_pred)),
                "tp": int(tp),
                "fp": int(fp),
                "fn": int(fn),
                "tn": int(tn),
            }
        )
    return pd.DataFrame(rows)


def thresholds_for_target_recall(y_true, y_proba, target_recall: float) -> pd.DataFrame:
    """The highest thresholds that still reach a required sensitivity.

    In a screening context the question is "how many people must I flag to catch
    X% of the people who will develop CHD?", which is a different question from
    "which threshold maximises F1?".
    """
    table = threshold_table(y_true, y_proba, thresholds=np.round(np.arange(0.01, 1.0, 0.01), 2))
    reachable = table[table["recall"] >= target_recall]
    if reachable.empty:
        return table.iloc[0:0]
    # Keep the *single* most selective threshold that still meets the target.
    best = reachable.sort_values(["precision", "threshold"], ascending=False).head(1)
    return best
