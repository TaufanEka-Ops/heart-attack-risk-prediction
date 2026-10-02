"""Loading and validating the Framingham subset.

This module never imputes, never scales, and never splits.  It only answers
"is the file the file I think it is?" so that every downstream mistake is a
modelling mistake rather than a data mistake.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import (
    EXPECTED_COLUMNS,
    EXPECTED_MISSING,
    EXPECTED_N_ROWS,
    PLAUSIBLE_RANGES,
    RAW_CSV,
    TARGET,
)

__all__ = [
    "load_raw",
    "load_dataset",
    "validate_schema",
    "missingness_table",
    "data_contract",
]


def load_raw(path=None) -> pd.DataFrame:
    """Read the raw CSV with no mutation whatsoever."""
    path = RAW_CSV if path is None else path
    return pd.read_csv(path)


def validate_schema(df: pd.DataFrame) -> None:
    """Raise ValueError if `df` violates the data contract.

    Checks, in order: exact column set and order, no duplicate rows, every
    column numeric, the target binary, no negative values in non-negative
    fields, values inside plausible ranges, and no infinity.
    """
    problems: list[str] = []

    if list(df.columns) != EXPECTED_COLUMNS:
        problems.append(
            f"columns differ from the contract.\n  expected: {EXPECTED_COLUMNS}\n"
            f"  actual:   {list(df.columns)}"
        )

    if len(df) != EXPECTED_N_ROWS:
        problems.append(f"row count is {len(df)}, expected {EXPECTED_N_ROWS}")

    dupes = int(df.duplicated().sum())
    if dupes:
        problems.append(f"{dupes} fully duplicated rows")

    if TARGET in df.columns:
        observed = set(pd.unique(df[TARGET].dropna()))
        if not observed <= {0, 1}:
            problems.append(f"target {TARGET} is not binary, observed {sorted(observed)}")

    non_numeric = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
    if non_numeric:
        problems.append(f"non-numeric columns: {non_numeric}")

    for col, (low, high) in PLAUSIBLE_RANGES.items():
        if col not in df.columns:
            continue
        series = df[col].dropna()
        if series.empty:
            continue
        if series.min() < low or series.max() > high:
            problems.append(
                f"{col} out of plausible range [{low}, {high}]: "
                f"observed [{series.min()}, {series.max()}]"
            )

    numeric = df.select_dtypes(include="number")
    if numeric.size:
        infinity = int(np.isinf(numeric.to_numpy(dtype="float64")).sum())
        if infinity:
            problems.append(f"{infinity} infinite values")

    if problems:
        raise ValueError("data contract violated:\n - " + "\n - ".join(problems))


def load_dataset(path=None) -> tuple[pd.DataFrame, pd.Series]:
    """Return ``(X, y)`` with X a DataFrame of raw, un-imputed features.

    Missing values are intentionally preserved.  Imputation happens inside the
    model pipeline, on training folds only — see :mod:`src.preprocess`.
    """
    df = load_raw(path)
    validate_schema(df)
    y = df[TARGET].astype("int64")
    X = df.drop(columns=[TARGET]).copy()
    return X, y


def missingness_table(df: pd.DataFrame) -> pd.DataFrame:
    """Per-column missing counts and percentages, sorted by severity."""
    counts = df.isna().sum()
    table = pd.DataFrame(
        {
            "missing": counts.astype(int),
            "pct_missing": (counts / len(df) * 100).round(2),
        }
    )
    table = table[table["missing"] > 0].sort_values("missing", ascending=False)
    return table


def data_contract(df: pd.DataFrame) -> dict:
    """Machine-readable summary of the file, used by ``python -m src --check``."""
    missing = {c: int(n) for c, n in df.isna().sum().items() if n > 0}
    positives = int(df[TARGET].sum())
    return {
        "n_rows": int(len(df)),
        "n_columns": int(df.shape[1]),
        "positive_class": positives,
        "negative_class": int(len(df) - positives),
        "positive_rate": round(positives / len(df), 4),
        "missing": missing,
        "missing_matches_expected": missing == EXPECTED_MISSING,
        "duplicate_rows": int(df.duplicated().sum()),
    }
