# Data

`raw/framingham.csv` — the Framingham Heart Study 10-year CHD subset.

| | |
|---|---|
| Rows | 4,238 participants |
| Columns | 16 (15 features + target) |
| Target | `TenYearCHD`, binary; 644 positives (15.20%) |
| Missing cells | 645 (0.95% of the matrix), across 7 columns |
| SHA-256 | `40921748dbdf6587045b9822414199a1ae3349358b63b975ab2208aa3cccc45b` |
| MD5 | `d63c4429755949141d1ccea1ebaf997a` |

The file is included **unmodified**. Its hashes are recorded in
`results/manifest.json`, which `scripts/06_manifest.py` regenerates — so a
reviewer can confirm the bytes are the same ones the results were produced from.

## Columns

| Column | Type | Notes |
|---|---|---|
| male | binary 0/1 | |
| age | continuous | years at exam |
| education | ordinal 1–4 | 105 missing |
| currentSmoker | binary 0/1 | |
| cigsPerDay | continuous | 0 for non-smokers; 29 missing |
| BPMeds | binary 0/1 | on blood-pressure medication; 53 missing |
| prevalentStroke | binary 0/1 | |
| prevalentHyp | binary 0/1 | |
| diabetes | binary 0/1 | |
| totChol | continuous | mg/dL; 50 missing |
| sysBP | continuous | mmHg |
| diaBP | continuous | mmHg |
| BMI | continuous | 19 missing |
| heartRate | continuous | bpm; 1 missing |
| glucose | continuous | mg/dL; 388 missing (9.16%) |
| **TenYearCHD** | **binary 0/1** | **target** |

## Missing values

| Column | Missing | % |
|---|---|---|
| glucose | 388 | 9.16 |
| education | 105 | 2.48 |
| BPMeds | 53 | 1.25 |
| totChol | 50 | 1.18 |
| cigsPerDay | 29 | 0.68 |
| BMI | 19 | 0.45 |
| heartRate | 1 | 0.02 |

The loader (`src/data.py`) **does not impute**. It validates the file and returns
the frame with missing values intact, so imputation happens inside the model
pipeline on training folds only. See `docs/methodology.md` for why, and
`docs/methodology.md#missingness-is-not-obviously-random` for the evidence that
missing-at-random is an assumption here rather than a fact.

## Provenance and licence

This dataset is a widely circulated teaching subset derived from the
NHLBI-funded Framingham Heart Study, itself distributed under a data-use
agreement. It is included here for reproducibility only; check the upstream
terms before redistributing it. The MIT licence in the repository root covers the
code, not the data.
