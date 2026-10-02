"""Tiny helpers to make script runs self-documenting.

Every script writes a full transcript to ``results/transcripts/<name>.log`` so
the committed logs and the committed CSV are produced by the same run — a
reviewer can diff the two.
"""

from __future__ import annotations

import io
import sys
from contextlib import contextmanager

import pandas as pd

from .config import PROJECT_ROOT, TRANSCRIPTS_DIR

__all__ = ["transcript", "print_table", "save_csv"]


@contextmanager
def transcript(name: str):
    """Mirror everything printed inside the block to ``results/transcripts/``."""
    TRANSCRIPTS_DIR.mkdir(parents=True, exist_ok=True)
    path = TRANSCRIPTS_DIR / f"{name}.log"
    buffer = io.StringIO()
    real_stdout = sys.stdout

    class _Tee:
        def write(self, text):
            real_stdout.write(text)
            buffer.write(text)

        def flush(self):
            real_stdout.flush()

    sys.stdout = _Tee()
    try:
        yield path
    finally:
        sys.stdout = real_stdout
        path.write_text(buffer.getvalue(), encoding="utf-8")


def print_table(df: pd.DataFrame, float_format: str = "{:.4f}", title: str | None = None) -> None:
    """Print a DataFrame without pandas' index clutter."""
    if title:
        print(title)
        print("-" * len(title))
    with pd.option_context("display.width", 200, "display.max_columns", 50):
        print(df.to_string(index=False, float_format=lambda v: float_format.format(v)))
    print()


def save_csv(df: pd.DataFrame, path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"wrote {path.relative_to(PROJECT_ROOT)}")
