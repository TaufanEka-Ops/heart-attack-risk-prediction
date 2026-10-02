"""Pytest root configuration.

Having a conftest.py at the repository root puts the root on ``sys.path`` (pytest
prepends the conftest's directory), so ``import src`` works regardless of the
directory pytest is invoked from.  The explicit insert below makes the intent
obvious and keeps things working if the import mode is changed.
"""

from __future__ import annotations

import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="session")
def dataset():
    """The real dataset, loaded once for the whole test session."""
    from src.data import load_dataset

    return load_dataset()


@pytest.fixture(scope="session")
def raw_frame():
    from src.data import load_raw

    return load_raw()
