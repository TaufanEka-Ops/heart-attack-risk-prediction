"""Regression locks on the numbers quoted in README.md.

These tests read the CSVs that ``scripts/03``–``05`` produced and assert the
headline claims.  They are the reason the README cannot drift away from the
artifacts: if a modelling change moves a number, a test fails and the prose has
to be updated too.

They skip (rather than fail) when ``results/`` has not been generated yet, so a
fresh clone can run ``pytest`` before ``make all``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.config import PROJECT_ROOT, RESULTS_DIR

CV_CSV = RESULTS_DIR / "metrics_cv.csv"
HOLDOUT_CSV = RESULTS_DIR / "metrics_holdout.csv"
CALIB_CSV = RESULTS_DIR / "calibration_summary.csv"
OOF_CSV = RESULTS_DIR / "oof_probabilities.csv"
IMPUTATION_CSV = RESULTS_DIR / "leakage_audit_imputation.csv"
SCALER_CSV = RESULTS_DIR / "leakage_audit_scaler.csv"
THRESHOLD_CSV = RESULTS_DIR / "threshold_analysis.csv"

needs_results = pytest.mark.skipif(
    not CV_CSV.exists(), reason="run `make all` first to generate results/"
)


@pytest.fixture(scope="module")
def cv_table():
    return pd.read_csv(CV_CSV)


@pytest.fixture(scope="module")
def calibration_table():
    return pd.read_csv(CALIB_CSV)


# --------------------------------------------------------------------------- #
# Shape of the artifacts                                                       #
# --------------------------------------------------------------------------- #
@needs_results
def test_cross_validation_covers_every_model_and_metric(cv_table):
    assert len(cv_table) == 7
    for metric in ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "brier"]:
        assert f"{metric}_mean" in cv_table.columns
        assert f"{metric}_std" in cv_table.columns
    assert (cv_table["n_folds"] == 5).all()


# --------------------------------------------------------------------------- #
# The headline claims in the README                                            #
# --------------------------------------------------------------------------- #
@needs_results
def test_majority_class_baseline_accuracy(cv_table):
    """'Predicting no CHD for everyone scores 84.8% accuracy.'"""
    dummy = cv_table.set_index("model").loc["Dummy (majority class)"]
    assert dummy["accuracy_mean"] == pytest.approx(0.8480, abs=0.0005)
    assert dummy["recall_mean"] == 0.0
    assert dummy["pr_auc_mean"] == pytest.approx(0.1520, abs=0.001)


@needs_results
def test_logistic_regression_accuracy_is_barely_above_the_dummy(cv_table):
    """The accuracy trap: better than 'always no' by 0.7 of a point."""
    table = cv_table.set_index("model")
    gain = table.loc["Logistic Regression", "accuracy_mean"] - table.loc[
        "Dummy (majority class)", "accuracy_mean"
    ]
    assert 0.0 < gain < 0.01


@needs_results
def test_logistic_regression_barely_finds_the_positive_class(cv_table):
    """'82.5% precision but catches 8% of the people who go on to develop CHD.'"""
    lr = cv_table.set_index("model").loc["Logistic Regression"]
    assert lr["recall_mean"] == pytest.approx(0.0807, abs=0.005)
    assert lr["recall_mean"] < 0.12
    assert lr["precision_mean"] == pytest.approx(0.7099, abs=0.01)


@needs_results
def test_best_model_and_its_pr_auc(cv_table):
    best = cv_table.sort_values("pr_auc_mean", ascending=False).iloc[0]
    assert best["model"] == "Logistic Regression"
    assert best["pr_auc_mean"] == pytest.approx(0.3466, abs=0.01)
    assert best["roc_auc_mean"] == pytest.approx(0.7250, abs=0.01)


@needs_results
def test_discrimination_ceiling_is_modest_not_stellar(cv_table):
    """Honesty guard: if some model starts scoring 0.9 ROC-AUC, something leaked."""
    assert cv_table["roc_auc_mean"].max() < 0.80
    assert cv_table["pr_auc_mean"].max() < 0.50


@needs_results
def test_every_model_beats_the_no_skill_baseline(cv_table):
    baseline = cv_table.set_index("model").loc["Dummy (majority class)", "pr_auc_mean"]
    for _, row in cv_table.iterrows():
        if row["model"] == "Dummy (majority class)":
            continue
        assert row["pr_auc_mean"] > baseline
        assert row["roc_auc_mean"] > 0.5


@needs_results
def test_models_are_separated_by_less_than_the_fold_noise(cv_table):
    """The honest conclusion: model choice barely matters here."""
    real = cv_table[cv_table["model"] != "Dummy (majority class)"]
    spread = real["pr_auc_mean"].max() - real["pr_auc_mean"].min()
    typical_std = real["pr_auc_std"].mean()
    assert spread < 2.5 * typical_std


# --------------------------------------------------------------------------- #
# Calibration                                                                  #
# --------------------------------------------------------------------------- #
@needs_results
def test_plain_logistic_regression_is_calibrated_in_the_large(calibration_table):
    row = calibration_table.set_index("model").loc["Logistic Regression"]
    assert row["mean_predicted"] == pytest.approx(row["observed_rate"], abs=0.005)
    assert row["calibration_ratio"] == pytest.approx(1.0, abs=0.02)


@needs_results
def test_balanced_logistic_regression_inflates_probabilities_threefold(calibration_table):
    table = calibration_table.set_index("model")
    balanced = table.loc["Logistic Regression (balanced)"]
    plain = table.loc["Logistic Regression"]

    assert balanced["calibration_ratio"] > 2.5
    assert balanced["brier"] > 1.5 * plain["brier"]
    assert balanced["ece_quantile_10"] > 10 * plain["ece_quantile_10"]


@needs_results
def test_balancing_did_not_improve_the_ranking(calibration_table):
    """Spearman ~0.998: same ordering, different probabilities.

    This is the core finding — class_weight='balanced' bought an operating
    point, not discrimination.
    """
    table = calibration_table.set_index("model")
    assert table.loc["Logistic Regression (balanced)", "spearman_vs_reference"] > 0.99


@needs_results
def test_balanced_and_plain_logistic_regression_have_the_same_roc_auc(cv_table):
    table = cv_table.set_index("model")
    difference = abs(
        table.loc["Logistic Regression", "roc_auc_mean"]
        - table.loc["Logistic Regression (balanced)", "roc_auc_mean"]
    )
    assert difference < 0.005


# --------------------------------------------------------------------------- #
# Threshold analysis                                                           #
# --------------------------------------------------------------------------- #
@needs_results
def test_default_threshold_misses_most_of_the_positive_class():
    table = pd.read_csv(THRESHOLD_CSV)
    row = table.loc[(table["threshold"] - 0.5).abs().idxmin()]
    assert row["recall"] < 0.12
    assert row["flagged_rate"] < 0.03

    # And there exists a much lower threshold with a much better F1.
    assert table["f1"].max() > 2.5 * row["f1"]


@needs_results
def test_reaching_seventy_percent_sensitivity_requires_flagging_a_large_share():
    table = pd.read_csv(RESULTS_DIR / "thresholds_for_target_recall.csv")
    assert len(table) == 1
    row = table.iloc[0]
    assert row["recall"] >= 0.70
    assert row["flagged_rate"] > 0.35
    assert row["precision"] < 0.35


# --------------------------------------------------------------------------- #
# Leakage audit                                                                #
# --------------------------------------------------------------------------- #
@needs_results
def test_imputation_audit_reports_158_out_of_domain_cells():
    table = pd.read_csv(IMPUTATION_CSV)
    assert int(table["cells_filled_out_of_domain"].sum()) == 158
    values = table.set_index("column")["cells_filled_out_of_domain"]
    assert values["education"] == 105
    assert values["BPMeds"] == 53


@needs_results
def test_scaler_audit_shows_full_data_statistics_differ_from_train_only():
    table = pd.read_csv(SCALER_CSV)
    assert (table["mean_abs_diff"] > 0).all()
    # totChol moved the most in absolute terms (mg/dL); check it is the top row
    assert table.iloc[0]["column"] == "totChol"
    assert table.iloc[0]["mean_abs_diff"] == pytest.approx(0.4661, abs=0.01)


@needs_results
def test_out_of_fold_probabilities_are_one_row_per_participant(oof_table):
    from src.config import EXPECTED_N_ROWS

    assert len(oof_table) == EXPECTED_N_ROWS
    assert int(oof_table["y_true"].sum()) == 644
    for column in [c for c in oof_table.columns if c != "y_true"]:
        assert oof_table[column].between(0, 1).all()


@pytest.fixture(scope="module")
def oof_table():
    if not OOF_CSV.exists():
        pytest.skip("run `make all` first to generate results/")
    return pd.read_csv(OOF_CSV)


# --------------------------------------------------------------------------- #
# README integrity                                                             #
# --------------------------------------------------------------------------- #
def test_readme_exists_and_quotes_the_produced_numbers():
    """Every headline figure in the README must be one the scripts produced."""
    readme = PROJECT_ROOT / "README.md"
    assert readme.exists(), "README.md is part of the deliverable"
    text = readme.read_text(encoding="utf-8")

    # These strings are all present in results/*.csv as produced by the scripts.
    for required in [
        "4,238",   # participants (thousands separator, as the README renders it)
        "15.2",    # positive rate
        "0.848",   # majority-class accuracy
        "0.081",   # cross-validated recall of the plain logistic regression at 0.5
        "0.3466",  # best cross-validated PR-AUC
        "0.7250",  # best cross-validated ROC-AUC
        "158",     # cells the original imputation rule filled out of domain
    ]:
        assert required in text, f"README should quote the produced value {required}"


def test_readme_does_not_promise_things_the_code_cannot_do():
    text = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8").lower()
    for overclaim in ["state of the art", "clinical grade", "production ready", "deploy"]:
        assert overclaim not in text, f"README contains the overclaim: {overclaim!r}"


def test_results_manifest_lists_generated_files():
    """A JSON manifest keeps the report honest about what was written."""
    manifest_path = RESULTS_DIR / "manifest.json"
    if not manifest_path.exists():
        pytest.skip("run `make all` first to generate results/manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["n_models"] == 7
    assert Path(manifest["raw_csv"]).exists()
    assert manifest["raw_csv_sha256"]
