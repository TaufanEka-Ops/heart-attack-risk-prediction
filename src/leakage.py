"""A faithful re-implementation of the *original* notebook's preprocessing.

This module exists so the bug can be measured instead of merely asserted.  It
is imported by ``scripts/02_leakage_audit.py`` and the leakage tests; nothing in
the production path (``src.models``, ``src.preprocess``) ever imports it.

The two defects, verbatim from the original notebook:

    data['education'].fillna(data['education'].mean(), inplace=True)      # x7
    ...
    scaler = StandardScaler()
    scaler.fit(X)                          # full matrix, before the split
    X_train, X_test, y_train, y_test = train_test_split(scaled_data, y, ...)
"""

from __future__ import annotations

import pandas as pd
from sklearn.preprocessing import StandardScaler

__all__ = ["leaky_preprocess", "leaky_pipeline"]


def leaky_preprocess(X_full: pd.DataFrame, fillna_columns=None):
    """Mean-impute every column, then scale using statistics of the whole frame.

    Returns ``(scaled_df, artifacts)`` where ``artifacts`` exposes the fitted
    scaler and the per-column means so a test can show they were computed on
    data that ends up in the test set.
    """
    X = X_full.copy()
    columns = list(X.columns) if fillna_columns is None else list(fillna_columns)
    means = X[columns].mean()
    for col in columns:
        X[col] = X[col].fillna(means[col])

    scaler = StandardScaler().fit(X)
    scaled = pd.DataFrame(scaler.transform(X), columns=X.columns, index=X.index)
    return scaled, {"scaler": scaler, "fill_means": means}


def leaky_pipeline(estimator):
    """A pipeline that leaks: the preprocessor is fitted on everything it sees.

    Historically this is what ``make_pipeline(MinMaxScaler(), SVC())`` did in the
    original notebook once the scaling had already happened on the full frame.
    Kept only for the audit script.
    """
    from .preprocess import build_pipeline

    return build_pipeline(estimator)
