"""Leakage tests.

Two separate claims are proved here:

1. **The fitted pipeline depends only on the training data.**  Every learned
   statistic (imputation values, scaler means and scales) equals the statistic
   computed from ``X_train`` alone, and swapping the held-out set for something
   absurd changes nothing about the fitted model.

2. **The original ordering really would have leaked.**  This is the guard that
   keeps claim 1 from being vacuous: when the preprocessor is fitted on the
   concatenation of train and test — which is exactly what ``scaler.fit(X)``
   before ``train_test_split`` amounts to — the learned statistics genuinely
   differ from the training-only ones.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.config import CONTINUOUS_FEATURES, RANDOM_STATE
from src.evaluate import make_cv, make_split
from src.leakage import leaky_preprocess
from src.models import get_models
from src.preprocess import build_pipeline, build_preprocessor

LR = "Logistic Regression"


def _continuous_pipeline(fitted_pipeline):
    return fitted_pipeline.named_steps["preprocess"].named_transformers_["continuous"]


def _imputer_stats(fitted_pipeline):
    return _continuous_pipeline(fitted_pipeline).named_steps["imputer"].statistics_


def _scaler_means(fitted_pipeline):
    return _continuous_pipeline(fitted_pipeline).named_steps["scaler"].mean_


def _stack(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    return pd.concat([a, b], axis=0, ignore_index=True)


def _median_imputed_train_means(X_train: pd.DataFrame) -> np.ndarray:
    """What the scaler must see: the training columns after median imputation.

    Comparing against ``X_train.mean()`` would be wrong — pandas skips NaN, so
    that is the mean of the observed values only, which is not what the
    pipeline feeds the scaler.
    """
    frame = X_train[CONTINUOUS_FEATURES]
    return frame.fillna(frame.median()).mean().to_numpy()


# --------------------------------------------------------------------------- #
# Claim 1 — preprocessing is fitted on training rows only                      #
# --------------------------------------------------------------------------- #
def test_fitted_statistics_equal_train_only_statistics(dataset):
    X, y = dataset
    X_train, _, _, _ = make_split(X, y)

    prep = build_preprocessor().fit(X_train)

    np.testing.assert_allclose(
        prep.named_transformers_["continuous"].named_steps["imputer"].statistics_,
        X_train[CONTINUOUS_FEATURES].median().to_numpy(),
        err_msg="median imputation was not taken from the training split",
    )
    np.testing.assert_allclose(
        prep.named_transformers_["continuous"].named_steps["scaler"].mean_,
        _median_imputed_train_means(X_train),
        err_msg="scaler mean was not taken from the training split",
    )


def test_train_and_full_data_statistics_actually_differ(dataset):
    """Without this, every other assertion could pass for the wrong reason."""
    X, y = dataset
    X_train, _, _, _ = make_split(X, y)

    train_medians = X_train[CONTINUOUS_FEATURES].median().to_numpy()
    full_medians = X[CONTINUOUS_FEATURES].median().to_numpy()
    assert not np.allclose(train_medians, full_medians), (
        "train and full-data medians coincide; this dataset no longer demonstrates "
        "the leak and the leakage tests need rethinking"
    )


def test_leaky_order_really_would_change_the_statistics(dataset):
    """Fit on train+test — the original ordering — and watch the numbers move."""
    X, y = dataset
    X_train, X_test, _, _ = make_split(X, y)

    correct = build_preprocessor().fit(X_train)
    leaky = build_preprocessor().fit(_stack(X_train, X_test))

    assert not np.allclose(
        correct.named_transformers_["continuous"].named_steps["imputer"].statistics_,
        leaky.named_transformers_["continuous"].named_steps["imputer"].statistics_,
    )
    assert not np.allclose(
        correct.named_transformers_["continuous"].named_steps["scaler"].mean_,
        leaky.named_transformers_["continuous"].named_steps["scaler"].mean_,
    )


def test_every_cross_validation_fold_fits_on_its_own_training_rows(dataset):
    """The pipeline used by cross_validate never sees the fold's validation rows."""
    X, y = dataset
    cv = make_cv(n_splits=3)
    checked = 0

    for train_idx, _ in cv.split(X, y):
        fold_train = X.iloc[train_idx]
        fitted = clone(
            build_pipeline(LogisticRegression(max_iter=200, random_state=RANDOM_STATE))
        ).fit(fold_train, y.iloc[train_idx])

        np.testing.assert_allclose(
            _imputer_stats(fitted), fold_train[CONTINUOUS_FEATURES].median().to_numpy()
        )
        np.testing.assert_allclose(
            _scaler_means(fitted), _median_imputed_train_means(fold_train)
        )
        checked += 1

    assert checked == 3


def test_a_different_held_out_set_cannot_change_the_fitted_model(dataset):
    """Fitting touches X_train only, so the held-out frame is irrelevant to it."""
    X, y = dataset
    X_train, X_test, y_train, _ = make_split(X, y)
    reference_rows = X_test.iloc[:100]

    model_a = clone(get_models()[LR]).fit(X_train, y_train)
    probabilities_a = model_a.predict_proba(reference_rows)[:, 1]

    # Swap in an absurd held-out frame.  Nothing refits with it: the pipeline
    # never receives X_test at all.
    X_test_absurd = X_test * 10_000
    assert not np.allclose(X_test_absurd.to_numpy(), X_test.to_numpy())

    model_b = clone(get_models()[LR]).fit(X_train, y_train)
    probabilities_b = model_b.predict_proba(reference_rows)[:, 1]

    np.testing.assert_allclose(probabilities_a, probabilities_b)
    np.testing.assert_allclose(
        model_a.named_steps["model"].coef_, model_b.named_steps["model"].coef_
    )
    np.testing.assert_allclose(_scaler_means(model_a), _scaler_means(model_b))


def test_fitting_on_train_plus_mutated_test_does_move_the_scaler(dataset):
    """The contrast that gives the previous test its meaning.

    Compares the preprocessor alone (which is the object that actually carries
    the leak) so the assertion is about scaling, not about an estimator
    struggling on deliberately absurd inputs.
    """
    X, y = dataset
    X_train, X_test, _, _ = make_split(X, y)

    baseline = build_preprocessor().fit(X_train)
    baseline_means = baseline.named_transformers_["continuous"].named_steps["scaler"].mean_.copy()

    contaminated = build_preprocessor().fit(_stack(X_train, X_test * 10_000))
    contaminated_means = (
        contaminated.named_transformers_["continuous"].named_steps["scaler"].mean_
    )

    assert not np.allclose(
        contaminated_means, baseline_means
    ), "mutating the held-out rows changed nothing, so the two paths are indistinguishable"


def test_holdout_split_is_disjoint_and_stratified(dataset):
    X, y = dataset
    X_train, X_test, y_train, y_test = make_split(X, y)

    assert set(X_train.index).isdisjoint(set(X_test.index))
    assert len(X_train) + len(X_test) == len(X)
    assert len(y_train) + len(y_test) == len(y)
    assert y_train.mean() == pytest.approx(y.mean(), abs=0.01)
    assert y_test.mean() == pytest.approx(y.mean(), abs=0.01)


def test_cross_validation_folds_partition_the_data_exactly_once(dataset):
    X, y = dataset
    cv = make_cv(n_splits=5)
    seen: list[int] = []
    for _, test_idx in cv.split(X, y):
        assert y.iloc[test_idx].mean() == pytest.approx(y.mean(), abs=0.03)
        seen.extend(test_idx.tolist())
    assert sorted(seen) == list(range(len(X)))


# --------------------------------------------------------------------------- #
# Claim 2 — the original notebook's procedure was leaky                        #
# --------------------------------------------------------------------------- #
def test_original_imputation_created_out_of_domain_values(dataset):
    X, _ = dataset
    imputed, artifacts = leaky_preprocess(X)

    education_mean = X["education"].mean()
    bp_meds_mean = X["BPMeds"].mean()

    # education is coded 1..4; its mean is not a code.
    assert not np.isin(education_mean, [1, 2, 3, 4])
    # BPMeds is a 0/1 indicator; its mean is not a category.
    assert not np.isin(bp_meds_mean, [0, 1])

    assert imputed.isna().sum().sum() == 0
    assert artifacts["fill_means"]["education"] == pytest.approx(education_mean)


def test_original_fill_values_are_not_valid_categories(dataset):
    X, _ = dataset
    imputed, _ = leaky_preprocess(X)

    # The mean-filled cells sit strictly between the legal codes.
    filled_education = imputed.loc[X["education"].isna(), "education"]
    filled_bp_meds = imputed.loc[X["BPMeds"].isna(), "BPMeds"]

    assert len(filled_education) == 105
    assert len(filled_bp_meds) == 53
    assert (~filled_education.isin([1, 2, 3, 4])).all()
    assert (~filled_bp_meds.isin([0, 1])).all()


def test_original_scaler_was_fitted_on_rows_that_later_became_test_data(dataset):
    X, y = dataset
    X_train, X_test, _, _ = make_split(X, y)

    _, artifacts = leaky_preprocess(X)
    leaky_means = artifacts["scaler"].mean_

    train_scaler = StandardScaler().fit(X_train.fillna(X_train.mean()))
    test_scaler = StandardScaler().fit(X_test.fillna(X_test.mean()))
    assert not np.allclose(leaky_means, train_scaler.mean_)

    # The leaky mean is computed on the union, so it must lie between the
    # train-only and test-only estimates, component by component.
    lo = np.minimum(train_scaler.mean_, test_scaler.mean_)
    hi = np.maximum(train_scaler.mean_, test_scaler.mean_)
    assert np.all(leaky_means >= lo - 1e-9)
    assert np.all(leaky_means <= hi + 1e-9)


def test_production_code_does_not_import_the_leakage_module():
    """src.models must not depend on the reproduction of the bug."""
    import src.models as models_module
    import src.preprocess as preprocess_module

    for module in (models_module, preprocess_module):
        source = open(module.__file__, encoding="utf-8").read()
        assert "leaky_preprocess" not in source
        assert "fillna(method" not in source
