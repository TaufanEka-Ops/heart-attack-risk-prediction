"""The data file is what the project claims it is."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.config import (
    EXPECTED_COLUMNS,
    EXPECTED_MISSING,
    EXPECTED_N_ROWS,
    FEATURE_COLUMNS,
    TARGET,
)
from src.data import data_contract, load_dataset, missingness_table, validate_schema


def test_column_order_matches_contract(raw_frame):
    assert list(raw_frame.columns) == EXPECTED_COLUMNS


def test_row_count(raw_frame):
    assert len(raw_frame) == EXPECTED_N_ROWS


def test_no_duplicate_participants(raw_frame):
    assert raw_frame.duplicated().sum() == 0


def test_missing_counts_are_exactly_as_documented(raw_frame):
    observed = {c: int(n) for c, n in raw_frame.isna().sum().items() if n > 0}
    assert observed == EXPECTED_MISSING


def test_target_is_binary_with_expected_prevalence(raw_frame):
    assert set(raw_frame[TARGET].unique()) == {0, 1}
    positives = int(raw_frame[TARGET].sum())
    assert positives == 644
    assert 0.14 < positives / len(raw_frame) < 0.16


def test_validate_schema_accepts_the_real_file(raw_frame):
    validate_schema(raw_frame)  # must not raise


def test_validate_schema_rejects_a_missing_column(raw_frame):
    with pytest.raises(ValueError, match="columns differ"):
        validate_schema(raw_frame.drop(columns=["glucose"]))


def test_validate_schema_rejects_extra_columns(raw_frame):
    polluted = raw_frame.copy()
    polluted["leaked_label"] = 1
    with pytest.raises(ValueError, match="columns differ"):
        validate_schema(polluted)


def test_validate_schema_rejects_a_non_binary_target(raw_frame):
    broken = raw_frame.copy()
    broken.loc[broken.index[0], TARGET] = 2
    with pytest.raises(ValueError, match="not binary"):
        validate_schema(broken)


def test_validate_schema_rejects_duplicated_rows(raw_frame):
    doubled = pd.concat([raw_frame, raw_frame.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicated"):
        validate_schema(doubled)


def test_validate_schema_rejects_implausible_values(raw_frame):
    broken = raw_frame.copy()
    broken.loc[broken.index[0], "sysBP"] = 9999.0
    with pytest.raises(ValueError, match="plausible range"):
        validate_schema(broken)


def test_load_dataset_preserves_missing_values(dataset):
    """The loader must NOT impute.  Imputation belongs inside the pipeline."""
    from src.config import EXPECTED_COLUMNS

    X, y = dataset
    # The loader keeps the file's own column order; the ColumnTransformer is what
    # regroups features later.
    assert list(X.columns) == EXPECTED_COLUMNS[:-1]
    assert sorted(X.columns) == sorted(FEATURE_COLUMNS)
    assert int(X.isna().sum().sum()) == sum(EXPECTED_MISSING.values())
    assert y.dtype == np.int64
    assert len(X) == len(y) == EXPECTED_N_ROWS


def test_loading_does_not_mutate_the_source_file(raw_frame):
    """load_dataset must not write anything back to disk."""
    from src.config import RAW_CSV

    before = RAW_CSV.read_bytes()
    load_dataset()
    assert RAW_CSV.read_bytes() == before


def test_missingness_table_is_sorted_and_correct(raw_frame):
    table = missingness_table(raw_frame)
    assert list(table.columns) == ["missing", "pct_missing"]
    assert table["missing"].is_monotonic_decreasing
    assert int(table["missing"].sum()) == sum(EXPECTED_MISSING.values())
    assert table.loc["glucose", "missing"] == 388
    # Percentages are computed against the full frame, not against non-null rows.
    assert table.loc["glucose", "pct_missing"] == pytest.approx(388 / 4238 * 100, abs=0.01)


def test_data_contract_summary(raw_frame):
    contract = data_contract(raw_frame)
    assert contract["n_rows"] == EXPECTED_N_ROWS
    assert contract["positive_class"] == 644
    assert contract["positive_rate"] == pytest.approx(0.152, abs=0.001)
    assert contract["missing_matches_expected"] is True
    assert contract["duplicate_rows"] == 0
