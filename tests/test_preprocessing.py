"""Preprocessing behaves per feature kind, and is a genuine re-fittable pipeline.

The headline defect in the original notebook was one imputation rule (the column
mean) applied to every column.  These tests pin the corrected behaviour.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.utils.validation import check_is_fitted

from src.config import BINARY_FEATURES, CONTINUOUS_FEATURES, FEATURE_COLUMNS, ORDINAL_FEATURES
from src.preprocess import (
    build_pipeline,
    build_preprocessor,
    imputed_columns,
    transformers_by_group,
)


def _imputer(preprocessor, group: str):
    return preprocessor.named_transformers_[group].named_steps["imputer"]


def _scaler(preprocessor):
    return preprocessor.named_transformers_["continuous"].named_steps["scaler"]


def test_feature_groups_partition_every_feature():
    groups = transformers_by_group()
    flat = [c for cols in groups.values() for c in cols]
    assert sorted(flat) == sorted(FEATURE_COLUMNS)
    assert len(flat) == len(set(flat)), "a feature must belong to exactly one group"


def test_preprocessor_starts_unfitted():
    with pytest.raises(Exception):
        check_is_fitted(build_preprocessor())


def test_imputation_strategies_differ_by_group(dataset):
    X, _ = dataset
    prep = build_preprocessor().fit(X)

    assert _imputer(prep, "continuous").strategy == "median"
    assert _imputer(prep, "binary").strategy == "most_frequent"
    assert _imputer(prep, "ordinal").strategy == "most_frequent"


def test_continuous_imputer_uses_the_median_not_the_mean(dataset):
    X, _ = dataset
    prep = build_preprocessor().fit(X)
    stats = _imputer(prep, "continuous").statistics_
    expected_medians = X[CONTINUOUS_FEATURES].median().to_numpy()
    np.testing.assert_allclose(stats, expected_medians)

    means = X[CONTINUOUS_FEATURES].mean().to_numpy()
    # For a skewed column like glucose the two differ, so this is not a vacuous check.
    assert not np.allclose(stats, means)


def test_categorical_columns_are_filled_with_in_domain_values(dataset):
    """The fix for defect A: no fractional education codes, no fractional BPMeds."""
    X, _ = dataset
    prep = build_preprocessor().fit(X)
    transformed = prep.transform(X)

    columns = list(prep.get_feature_names_out())
    education = transformed[:, columns.index("education")]
    bp_meds = transformed[:, columns.index("BPMeds")]

    assert set(np.unique(education)) <= {1.0, 2.0, 3.0, 4.0}
    assert set(np.unique(bp_meds)) <= {0.0, 1.0}
    assert X["education"].isna().any(), "the test is pointless if nothing was missing"


def test_the_original_mean_rule_would_have_been_out_of_domain(dataset):
    """Contrast: what the original notebook actually produced."""
    X, _ = dataset
    education_mean = X["education"].mean()
    bp_meds_mean = X["BPMeds"].mean()
    assert education_mean not in (1.0, 2.0, 3.0, 4.0)
    assert bp_meds_mean not in (0.0, 1.0)


def test_output_has_no_missing_values(dataset):
    X, _ = dataset
    transformed = build_preprocessor().fit(X).transform(X)
    assert not np.isnan(np.asarray(transformed, dtype=float)).any()


def test_output_width_and_order(dataset):
    X, _ = dataset
    prep = build_preprocessor().fit(X)
    names = list(prep.get_feature_names_out())
    assert names == CONTINUOUS_FEATURES + BINARY_FEATURES + ORDINAL_FEATURES
    assert prep.transform(X).shape == (len(X), len(FEATURE_COLUMNS))


def test_continuous_columns_are_standardised_on_the_fitted_data(dataset):
    X, _ = dataset
    prep = build_preprocessor().fit(X)
    transformed = prep.transform(X)
    names = list(prep.get_feature_names_out())
    for col in CONTINUOUS_FEATURES:
        values = transformed[:, names.index(col)]
        assert abs(values.mean()) < 1e-9
        assert abs(values.std() - 1.0) < 1e-9


def test_binary_columns_are_left_unscaled(dataset):
    """Scaling a 0/1 column adds nothing; the values must survive untouched."""
    X, _ = dataset
    prep = build_preprocessor().fit(X)
    transformed = prep.transform(X)
    names = list(prep.get_feature_names_out())
    male = transformed[:, names.index("male")]
    np.testing.assert_allclose(male, X["male"].to_numpy())


def test_transforming_new_data_does_not_refit(dataset):
    X, _ = dataset
    prep = build_preprocessor().fit(X)
    stats_before = _imputer(prep, "continuous").statistics_.copy()
    scaler_before = _scaler(prep).mean_.copy()

    prep.transform(X.head(50))
    prep.transform(X.tail(10))

    np.testing.assert_allclose(_imputer(prep, "continuous").statistics_, stats_before)
    np.testing.assert_allclose(_scaler(prep).mean_, scaler_before)


def test_pipeline_wraps_estimator_after_preprocessing(dataset):
    from sklearn.linear_model import LogisticRegression

    X, y = dataset
    pipe = build_pipeline(LogisticRegression(max_iter=200))
    assert list(pipe.named_steps) == ["preprocess", "model"]
    pipe.fit(X, y)
    assert pipe.predict(X.head(5)).shape == (5,)


def test_imputed_columns_helper_lists_exactly_the_incomplete_columns():
    assert imputed_columns() == sorted(
        ["education", "cigsPerDay", "BPMeds", "totChol", "BMI", "heartRate", "glucose"]
    )
