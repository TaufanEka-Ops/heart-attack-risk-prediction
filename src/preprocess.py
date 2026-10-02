"""Preprocessing as a re-fittable sklearn pipeline.

Why this module exists
----------------------
The original notebook did two things in the wrong order:

1. ``data['glucose'].fillna(data['glucose'].mean(), inplace=True)`` — computed
   the mean over *all 4,238 rows*, including the rows that later became the test
   set.  Every imputed value therefore carries information from the test set.
2. ``scaler.fit(X)`` on the full feature matrix, *then* ``train_test_split``.
   The scaler's mean and standard deviation are functions of the test rows too.

Both are leakage.  The fix is structural, not cosmetic: imputation and scaling
are steps **inside** an sklearn ``Pipeline``.  ``cross_validate`` and
``train_test_split`` then guarantee — by construction — that the transformers
see only the training fold, because the pipeline is re-fitted per fold.

The three feature groups also get *different* imputation rules, because the
original's single "fill everything with the mean" rule is wrong for a 0/1
indicator (a mean of 0.03 for BPMeds is not a category) and wrong for an ordinal
1..4 education code (2.94 is not a level).
"""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import BINARY_FEATURES, CONTINUOUS_FEATURES, ORDINAL_FEATURES

__all__ = [
    "build_preprocessor",
    "build_pipeline",
    "transformers_by_group",
    "imputed_columns",
]


def build_preprocessor() -> ColumnTransformer:
    """Return an *unfitted* ColumnTransformer with one rule per feature group.

    continuous  -> median imputation, then standardisation
    binary 0/1  -> most-frequent imputation, left unscaled (scaling a 0/1 column
                   is a linear reparametrisation and buys nothing interpretable)
    education   -> most-frequent imputation (a modal category, not a mean)
    """
    continuous = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    binary = Pipeline(steps=[("imputer", SimpleImputer(strategy="most_frequent"))])
    ordinal = Pipeline(steps=[("imputer", SimpleImputer(strategy="most_frequent"))])

    return ColumnTransformer(
        transformers=[
            ("continuous", continuous, CONTINUOUS_FEATURES),
            ("binary", binary, BINARY_FEATURES),
            ("ordinal", ordinal, ORDINAL_FEATURES),
        ],
        remainder="drop",  # anything unnamed is dropped, not silently passed through
        verbose_feature_names_out=False,
    )


def build_pipeline(estimator) -> Pipeline:
    """Wrap an estimator in ``preprocess -> estimator``.

    The returned object is a single mini-pipeline that a caller can hand to
    ``cross_validate``, ``GridSearchCV`` or ``train_test_split`` without having
    to remember anything about imputation order.
    """
    return Pipeline(steps=[("preprocess", build_preprocessor()), ("model", estimator)])


def transformers_by_group() -> dict[str, list[str]]:
    """The exact column -> group mapping, for tests and documentation."""
    return {
        "continuous": list(CONTINUOUS_FEATURES),
        "binary": list(BINARY_FEATURES),
        "ordinal": list(ORDINAL_FEATURES),
    }


def imputed_columns() -> list[str]:
    """Columns that actually contain missing values and therefore get imputed."""
    from .config import EXPECTED_MISSING

    return sorted(EXPECTED_MISSING)
