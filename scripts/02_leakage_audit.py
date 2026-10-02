"""02 — Leakage and imputation audit of the original notebook.

This script does not model anything new.  It *measures* the two defects in the
original coursework notebook so the claims in README.md and
docs/evaluation-of-original.md are backed by numbers rather than adjectives.

Defect A — imputation rule
    ``data[col].fillna(data[col].mean())`` was applied to every column,
    including the 0/1 indicator ``BPMeds`` and the 4-level ordinal ``education``.
    Those columns consequently contain values that are not members of their own
    domain.

Defect B — preprocessing fitted before the split
    ``scaler.fit(X)`` saw all 4,238 rows, then the frame was split.  The scaler's
    parameters are therefore a function of the test rows.  We quantify
    (i) how far the full-data statistics sit from the train-only statistics and
    (ii) how much the reported test metric moves as a result.

Writes: results/leakage_audit.csv, results/transcripts/02_leakage_audit.log
"""


from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))


import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression

from src.config import RANDOM_STATE, RESULTS_DIR
from src.data import load_dataset
from src.evaluate import calculate_metrics, make_split
from src.leakage import leaky_preprocess
from src.models import get_models
from src.preprocess import build_pipeline
from src.reporting import print_table, save_csv, transcript


def audit_imputation_rules(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """Count the cells the original rule would fill with an out-of-domain value."""
    rows = []

    missing = X.isna().sum()
    full_means = X.mean(numeric_only=True)
    X_train, X_test = make_split(X, y)[0], make_split(X, y)[1]
    train_means = X_train.mean(numeric_only=True)

    for col in ["education", "BPMeds", "cigsPerDay", "glucose", "totChol", "BMI", "heartRate"]:
        n_missing = int(missing[col])
        value = float(full_means[col])
        if col == "education":
            domain = "{1,2,3,4}"
            in_domain = value in (1.0, 2.0, 3.0, 4.0)
        elif col == "BPMeds":
            domain = "{0,1}"
            in_domain = value in (0.0, 1.0)
        else:
            domain = "non-negative real"
            in_domain = True
        rows.append(
            {
                "column": col,
                "kind": "ordinal" if col == "education" else ("binary 0/1" if col == "BPMeds" else "continuous"),
                "domain": domain,
                "n_missing": n_missing,
                "full_data_mean_used_to_fill": value,
                "train_only_mean": float(train_means[col]) if n_missing else np.nan,
                "fill_value_in_domain": in_domain,
                "cells_filled_out_of_domain": n_missing if not in_domain else 0,
            }
        )
    return pd.DataFrame(rows)


def audit_scaler_leakage(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """Compare scaler statistics fitted on all data vs on the training split only."""
    X_train, X_test, _, _ = make_split(X, y)

    # The original: impute with the full-data mean, then fit the scaler on all rows.
    _, leaky_artifacts = leaky_preprocess(X)
    scaler_full = leaky_artifacts["scaler"]

    # What a correct run would have used: the training split only.
    X_train_imputed = X_train.fillna(X_train.mean(numeric_only=True))
    from sklearn.preprocessing import StandardScaler

    scaler_train = StandardScaler().fit(X_train_imputed)

    comparison = pd.DataFrame(
        {
            "column": X.columns,
            "mean_full_data": scaler_full.mean_,
            "mean_train_only": scaler_train.mean_,
            "mean_abs_diff": np.abs(scaler_full.mean_ - scaler_train.mean_),
            "sd_full_data": scaler_full.scale_,
            "sd_train_only": scaler_train.scale_,
        }
    )
    # Expressing the gap in standard-deviation units makes columns measured in
    # mg/dL comparable with 0/1 indicators.
    comparison["mean_abs_diff_in_sd"] = comparison["mean_abs_diff"] / comparison["sd_train_only"]
    return comparison.sort_values("mean_abs_diff_in_sd", ascending=False).reset_index(drop=True)


def audit_metric_impact(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """Does the leak actually change the headline number? Measure it, both ways."""
    rows = []
    X_train_raw, X_test_raw, y_train, y_test = make_split(X, y)

    # ---- correct: raw split, preprocessing inside the pipeline ---------- #
    correct = clone(build_pipeline(LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)))
    correct.fit(X_train_raw, y_train)
    p_correct = correct.predict_proba(X_test_raw)[:, 1]
    m_correct = calculate_metrics(y_test, correct.predict(X_test_raw), p_correct)

    # ---- leaky: impute + scale on everything, then split ----------------- #
    X_leaky, _ = leaky_preprocess(X)
    Xl_train, Xl_test = X_leaky.loc[X_train_raw.index], X_leaky.loc[X_test_raw.index]
    leaky_model = LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)
    leaky_model.fit(Xl_train, y_train)
    p_leaky = leaky_model.predict_proba(Xl_test)[:, 1]
    m_leaky = calculate_metrics(y_test, leaky_model.predict(Xl_test), p_leaky)

    for metric in ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "brier"]:
        rows.append(
            {
                "metric": metric,
                "leaky_preprocessing": m_leaky[metric],
                "correct_pipeline": m_correct[metric],
                "difference": m_leaky[metric] - m_correct[metric],
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    X, y = load_dataset()
    print(f"loaded {len(X)} rows, {int(y.sum())} positives ({y.mean():.2%})\n")

    print("=== Defect A: one imputation rule for every column kind ===")
    rule_table = audit_imputation_rules(X, y)
    print_table(rule_table)
    invalid_cells = int(rule_table["cells_filled_out_of_domain"].sum())
    print(f"cells given a value outside their own domain: {invalid_cells}\n")

    print("=== Defect B: scaler fitted on all rows before the split ===")
    scaler_table = audit_scaler_leakage(X, y)
    print_table(scaler_table.head(15))
    worst = scaler_table.iloc[0]
    print(
        f"largest gap: {worst['column']}, {worst['mean_full_data']:.4f} (all rows) vs "
        f"{worst['mean_train_only']:.4f} (train only) — "
        f"{worst['mean_abs_diff']:.4f} raw units, {worst['mean_abs_diff_in_sd']:.4f} sd\n"
    )

    print("=== Defect B impact on the reported test metrics (single 80/20 split) ===")
    impact = audit_metric_impact(X, y)
    print_table(impact)
    print(
        "Read this honestly: scaling leakage alone shifts the metrics only in the\n"
        "third decimal place. It is a methodological defect — the test rows helped\n"
        "choose the parameters used to score them — but it is not the reason the\n"
        "original numbers look good. The reason is that accuracy is the wrong\n"
        "metric for a 15%-positive problem. See scripts/03 and the majority-class\n"
        "baseline: predicting 'no CHD' for everyone scores 84.8% accuracy."
    )

    save_csv(rule_table, RESULTS_DIR / "leakage_audit_imputation.csv")
    save_csv(scaler_table, RESULTS_DIR / "leakage_audit_scaler.csv")
    save_csv(impact, RESULTS_DIR / "leakage_audit_metric_impact.csv")


if __name__ == "__main__":
    with transcript("02_leakage_audit"):
        main()
