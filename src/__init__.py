"""Framingham 10-year CHD risk prediction — a rigorously specified rebuild.

The package is deliberately small and side-effect free: importing it must never
read a file, fit a model, or write to disk.  Scripts under ``scripts/`` are the
only place where I/O happens.
"""

from .config import (  # noqa: F401
    BINARY_FEATURES,
    CONTINUOUS_FEATURES,
    EXPECTED_COLUMNS,
    FEATURE_COLUMNS,
    ORDINAL_FEATURES,
    RANDOM_STATE,
    TARGET,
)

__version__ = "1.0.0"
