"""Metric helpers are computed correctly, not merely imported.

Every expected value below is worked out by hand in the docstring so the test is
a specification, not a snapshot of whatever scikit-learn happened to return.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.evaluate import (
    calculate_metrics,
    expected_calibration_error,
    make_cv,
    majority_baseline,
    threshold_table,
    thresholds_for_target_recall,
)

# y_true:    0 0 1 1
# y_pred:    0 1 1 1   -> tn=1 fp=1 tp=2 fn=0
# accuracy (1+2)/4 = 0.75, precision 2/3, recall 2/2 = 1.0, f1 0.8
Y_TRUE = np.array([0, 0, 1, 1])
Y_PRED = np.array([0, 1, 1, 1])
Y_PROBA = np.array([0.1, 0.6, 0.7, 0.4])


def test_calculate_metrics_matches_hand_computation():
    metrics = calculate_metrics(Y_TRUE, Y_PRED, Y_PROBA)
    assert metrics["accuracy"] == pytest.approx(0.75)
    assert metrics["precision"] == pytest.approx(2 / 3)
    assert metrics["recall"] == pytest.approx(1.0)
    assert metrics["f1"] == pytest.approx(0.8)


def test_roc_auc_matches_hand_computation():
    # positives score {0.7, 0.4}, negatives score {0.1, 0.6}; three of the four
    # positive/negative pairs are correctly ordered -> AUC = 0.75
    metrics = calculate_metrics(Y_TRUE, Y_PRED, Y_PROBA)
    assert metrics["roc_auc"] == pytest.approx(0.75)


def test_pr_auc_matches_hand_computation():
    # AP = sum (R_n - R_{n-1}) * P_n over score thresholds 0.7, 0.6, 0.4
    #    = 0.5*1.0 + 0*0.5 + 0.5*(2/3) = 0.8333
    metrics = calculate_metrics(Y_TRUE, Y_PRED, Y_PROBA)
    assert metrics["pr_auc"] == pytest.approx(2 / 3 * 0.5 + 0.5, abs=0.03)


def test_brier_score_matches_hand_computation():
    # mean of (0-0.1)^2, (0-0.6)^2, (1-0.7)^2, (1-0.4)^2 = 0.82/4 = 0.205
    metrics = calculate_metrics(Y_TRUE, Y_PRED, Y_PROBA)
    assert metrics["brier"] == pytest.approx(0.205)


def test_probability_metrics_are_nan_without_probabilities():
    metrics = calculate_metrics(Y_TRUE, Y_PRED, y_proba=None)
    assert np.isnan(metrics["roc_auc"])
    assert np.isnan(metrics["pr_auc"])
    assert np.isnan(metrics["brier"])


def test_perfect_predictions_score_one():
    y = np.array([0, 0, 1, 1])
    metrics = calculate_metrics(y, y, np.array([0.01, 0.02, 0.98, 0.99]))
    assert metrics["accuracy"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == 1.0
    assert metrics["roc_auc"] == 1.0
    assert metrics["brier"] < 0.01


def test_zero_division_is_handled_not_raised():
    y = np.array([0, 0, 1, 1])
    metrics = calculate_metrics(y, np.zeros(4, dtype=int), np.full(4, 0.5))
    assert metrics["precision"] == 0.0
    assert metrics["recall"] == 0.0
    assert metrics["f1"] == 0.0


# --------------------------------------------------------------------------- #
# Calibration                                                                  #
# --------------------------------------------------------------------------- #
def test_ece_is_near_zero_for_a_perfectly_calibrated_forecast():
    # 200 people predicted at 10%, of whom 20 develop CHD; 200 at 50%, of whom 100 do.
    proba = np.concatenate([np.full(200, 0.1), np.full(200, 0.5)])
    y = np.concatenate([np.zeros(180), np.ones(20), np.zeros(100), np.ones(100)])
    assert expected_calibration_error(y, proba, n_bins=2) == pytest.approx(0.0, abs=0.01)


def test_ece_is_large_for_a_badly_over_predicting_forecast():
    # everyone is told "90% risk"; the true rate is 10%.
    proba = np.full(200, 0.9)
    y = np.concatenate([np.zeros(180), np.ones(20)])
    assert expected_calibration_error(y, proba, n_bins=2) == pytest.approx(0.8, abs=0.01)


def test_ece_is_zero_for_a_single_constant_correct_bin():
    proba = np.full(50, 0.2)
    y = np.concatenate([np.ones(10), np.zeros(40)])
    assert expected_calibration_error(y, proba, n_bins=1) == pytest.approx(0.0, abs=1e-9)


# --------------------------------------------------------------------------- #
# Baselines and thresholds                                                     #
# --------------------------------------------------------------------------- #
def test_majority_baseline_on_the_real_prevalence():
    y = np.array([0] * 3594 + [1] * 644)
    baseline = majority_baseline(y)
    assert baseline["majority_class"] == 0
    assert baseline["accuracy_if_always_majority"] == pytest.approx(0.848, abs=0.001)
    assert baseline["recall_on_positives"] == 0.0
    assert baseline["pr_auc_of_constant"] == pytest.approx(0.152, abs=0.001)


def test_majority_baseline_flips_when_positives_dominate():
    y = np.array([0] * 10 + [1] * 90)
    baseline = majority_baseline(y)
    assert baseline["majority_class"] == 1
    assert baseline["recall_on_positives"] == 1.0


def test_default_threshold_row_is_identical_to_predict():
    y = np.array([0, 0, 1, 1, 1, 0])
    proba = np.array([0.2, 0.4, 0.45, 0.55, 0.9, 0.1])
    table = threshold_table(y, proba, thresholds=[0.5])
    row = table.iloc[0]
    # (proba >= 0.5) -> [0,0,0,1,1,0]  -> tp=2, fp=0, fn=1, tn=3
    assert (row["tp"], row["fp"], row["fn"], row["tn"]) == (2, 0, 1, 3)
    assert row["precision"] == pytest.approx(1.0)
    assert row["recall"] == pytest.approx(2 / 3)


def test_threshold_table_is_monotone_in_recall():
    rng = np.random.default_rng(0)
    y = rng.binomial(1, 0.15, 2000)
    proba = np.clip(0.15 + 0.4 * (y - 0.15) + rng.normal(0, 0.1, 2000), 0, 1)
    table = threshold_table(y, proba)
    # Raising the bar can only ever lose true positives.
    assert (table["recall"].diff().dropna() <= 1e-12).all()
    # And it can only ever flag fewer people.
    assert (table["flagged_rate"].diff().dropna() <= 1e-12).all()


def test_thresholds_for_target_recall_actually_reach_the_target():
    rng = np.random.default_rng(1)
    y = rng.binomial(1, 0.15, 3000)
    proba = np.clip(0.15 + 0.5 * (y - 0.15) + rng.normal(0, 0.12, 3000), 0, 1)
    table = thresholds_for_target_recall(y, proba, target_recall=0.70)
    assert not table.empty
    assert (table["recall"] >= 0.70).all()


# --------------------------------------------------------------------------- #
# Cross-validation splitter                                                    #
# --------------------------------------------------------------------------- #
def test_cv_is_stratified_and_reproducible():
    y = np.array([0] * 1000 + [1] * 200)
    X = np.zeros((len(y), 2))

    folds_a = list(make_cv(5).split(X, y))
    folds_b = list(make_cv(5).split(X, y))
    for (train_a, test_a), (train_b, test_b) in zip(folds_a, folds_b):
        np.testing.assert_array_equal(train_a, train_b)
        np.testing.assert_array_equal(test_a, test_b)

    for _, test_idx in folds_a:
        assert y[test_idx].mean() == pytest.approx(200 / 1200, abs=0.01)
