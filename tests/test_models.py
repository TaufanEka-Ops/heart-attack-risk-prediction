"""The model zoo is well-formed: every entry is a fresh, complete pipeline."""

from __future__ import annotations

import numpy as np
import pytest

from src.config import FEATURE_COLUMNS
from src.models import MODEL_ORDER, get_models


@pytest.fixture(scope="module")
def models():
    return get_models()


@pytest.fixture(scope="module")
def fitted_all(dataset):
    """Fit every model once on a stratified subsample; reuse across the tests."""
    from sklearn.model_selection import train_test_split

    X, y = dataset
    X_small, _, y_small, _ = train_test_split(
        X, y, train_size=800, stratify=y, random_state=0
    )
    return {name: pipe.fit(X_small, y_small) for name, pipe in get_models().items()}, X_small, y_small


def test_registry_has_exactly_the_documented_models(models):
    assert list(models.keys()) == MODEL_ORDER
    assert len(models) == 7


def test_every_model_is_a_complete_pipeline(models):
    for name, pipeline in models.items():
        steps = list(pipeline.named_steps)
        assert steps == ["preprocess", "model"], f"{name} has steps {steps}"


def test_preprocessing_is_part_of_the_model_not_applied_before_it(models):
    """The whole point: scaling and imputation are inside, so CV cannot leak."""
    for name, pipeline in models.items():
        preprocess = pipeline.named_steps["preprocess"]
        assert preprocess.transformers, f"{name} has no transformers"


def test_registry_returns_fresh_objects_every_call():
    first, second = get_models(), get_models()
    assert first.keys() == second.keys()
    for name in first:
        assert first[name] is not second[name], f"{name} is shared between calls"


def test_a_baseline_model_is_present():
    from sklearn.dummy import DummyClassifier

    dummy = get_models()["Dummy (majority class)"].named_steps["model"]
    assert isinstance(dummy, DummyClassifier)
    assert dummy.strategy == "prior"


def test_imbalance_aware_models_actually_set_class_weight():
    from sklearn.linear_model import LogisticRegression

    models = get_models()
    unbalanced = models["Logistic Regression"].named_steps["model"]
    balanced = models["Logistic Regression (balanced)"].named_steps["model"]

    assert isinstance(unbalanced, LogisticRegression)
    assert unbalanced.class_weight is None
    assert isinstance(balanced, LogisticRegression)
    assert balanced.class_weight == "balanced"


def test_every_model_fits_and_predicts(fitted_all):
    fitted, X_small, _ = fitted_all
    for name, model in fitted.items():
        predictions = model.predict(X_small)
        assert predictions.shape == (len(X_small),), name
        assert set(np.unique(predictions)) <= {0, 1}, name


def test_every_model_exposes_probabilities_in_the_unit_interval(fitted_all):
    fitted, X_small, _ = fitted_all
    for name, model in fitted.items():
        assert hasattr(model, "predict_proba"), f"{name} cannot produce probabilities"
        proba = model.predict_proba(X_small)[:, 1]
        assert np.all((proba >= 0) & (proba <= 1)), name
        assert np.isfinite(proba).all(), name


def test_models_are_deterministic_given_the_seed(fitted_all, dataset):
    """Two fits of the same model on the same rows must agree exactly."""
    from sklearn.base import clone

    fitted, X_small, y_small = fitted_all
    for name in ("Logistic Regression", "Random Forest (balanced)", "MLP (100,100,100)"):
        again = clone(get_models()[name]).fit(X_small, y_small)
        np.testing.assert_allclose(
            fitted[name].predict_proba(X_small)[:, 1],
            again.predict_proba(X_small)[:, 1],
        )


def test_the_mlp_uses_a_solver_that_actually_converges():
    """The original used solver='sgd' at lr 0.01, which trained worse than a GLM."""
    from sklearn.neural_network import MLPClassifier

    mlp = get_models()["MLP (100,100,100)"].named_steps["model"]
    assert isinstance(mlp, MLPClassifier)
    assert mlp.solver == "adam"
    assert mlp.early_stopping is True
    assert mlp.hidden_layer_sizes == (100, 100, 100)


def test_feature_columns_of_the_fitted_preprocessor(fitted_all):
    fitted, _, _ = fitted_all
    names = list(fitted["Logistic Regression"].named_steps["preprocess"].get_feature_names_out())
    assert names == FEATURE_COLUMNS
