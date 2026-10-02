"""06 — Provenance manifest.

Writes ``results/manifest.json``: the SHA-256 of the raw CSV, the versions of
every library that produced the numbers, and an inventory of the artifacts.  A
reviewer can re-run the hash check and know they are looking at the same data.

Writes: results/manifest.json, results/transcripts/06_manifest.log
"""

from __future__ import annotations

import hashlib
import json
import pathlib
import platform
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import sklearn

from src.config import N_SPLITS, PROJECT_ROOT, RAW_CSV, RANDOM_STATE, RESULTS_DIR
from src.data import data_contract, load_raw, validate_schema
from src.models import MODEL_ORDER
from src.reporting import transcript


def sha256_of(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    df = load_raw()
    validate_schema(df)
    contract = data_contract(df)

    artifacts = sorted(
        str(p.relative_to(PROJECT_ROOT))
        for p in RESULTS_DIR.rglob("*")
        if p.is_file() and p.name != "manifest.json"
    )

    manifest = {
        "project": "heart-attack-risk-prediction",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "raw_csv": str(RAW_CSV),
        "raw_csv_sha256": sha256_of(RAW_CSV),
        "raw_csv_md5": hashlib.md5(RAW_CSV.read_bytes()).hexdigest(),
        "n_rows": contract["n_rows"],
        "n_positives": contract["positive_class"],
        "positive_rate": contract["positive_rate"],
        "missing_cells": int(df.isna().sum().sum()),
        "n_models": len(MODEL_ORDER),
        "models": MODEL_ORDER,
        "protocol": {
            "cv": f"StratifiedKFold(n_splits={N_SPLITS}, shuffle=True)",
            "holdout": "train_test_split(test_size=0.2, stratify=y)",
            "random_state": RANDOM_STATE,
        },
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "artifacts": artifacts,
    }

    out = RESULTS_DIR / "manifest.json"
    out.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(manifest, indent=2))
    print()
    print(f"wrote {out.relative_to(PROJECT_ROOT)}  ({len(artifacts)} artifacts listed)")


if __name__ == "__main__":
    with transcript("06_manifest"):
        main()
