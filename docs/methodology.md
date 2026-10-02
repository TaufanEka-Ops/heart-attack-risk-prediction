# Methodology

This note states the protocol precisely enough that another analyst could
disagree with a specific decision rather than with the whole thing.

## 1. Data

`data/raw/framingham.csv` — 4,238 participants, 16 columns, unmodified from the
source. SHA-256 recorded in `results/manifest.json`. 644 participants (15.20%)
experienced coronary heart disease within ten years.

645 cells are missing across seven columns:

| Column | Missing | % |
|---|---|---|
| glucose | 388 | 9.16 |
| education | 105 | 2.48 |
| BPMeds | 53 | 1.25 |
| totChol | 50 | 1.18 |
| cigsPerDay | 29 | 0.68 |
| BMI | 19 | 0.45 |
| heartRate | 1 | 0.02 |

`src/data.py` validates the file and returns it **with missing values intact**.
Nothing in the loading path imputes. `validate_schema` asserts the column set and
order, the row count, absence of duplicate rows, a binary target, numeric dtypes,
no infinities, and plausible physical ranges.

### Missingness is not obviously random

`scripts/01_eda.py` cross-tabulates missingness against the outcome:

| Column | CHD rate when missing | CHD rate when present | n missing |
|---|---|---|---|
| BMI | 0.526 | 0.150 | 19 |
| BPMeds | 0.208 | 0.151 | 53 |
| totChol | 0.180 | 0.152 | 50 |
| glucose | 0.129 | 0.154 | 388 |
| cigsPerDay | 0.069 | 0.153 | 29 |
| education | 0.152 | 0.152 | 105 |

The BMI row is striking, but n = 19 and there is no mechanism story to go with
it, so it is reported and flagged rather than modelled. The point of the table is
that missing-at-random is an assumption here, not a fact — which is why the
"what I would do differently" section of the README argues for multiple
imputation instead of the median.

## 2. Preprocessing

Three feature groups, three rules, all inside one `ColumnTransformer`:

| Group | Columns | Rule |
|---|---|---|
| continuous | age, cigsPerDay, totChol, sysBP, diaBP, BMI, heartRate, glucose | median impute → standardise |
| binary | male, currentSmoker, BPMeds, prevalentStroke, prevalentHyp, diabetes | most-frequent impute, no scaling |
| ordinal | education (1–4) | most-frequent impute, no scaling |

Reasons for each choice:

- **Median, not mean, for continuous columns.** Glucose and totChol are skewed;
  the median is the more robust location estimate when 9% of a column is absent.
- **Most-frequent, not mean, for categoricals.** `education` takes values in
  {1,2,3,4}; its mean is 1.9789, which is not a level. `BPMeds` is 0/1; its mean
  is 0.0296, which is not a category. The original notebook imputed both with the
  mean, putting 158 cells outside their own column's domain.
- **Binary columns are not standardised.** A 0/1 column scaled to zero mean and
  unit variance is a linear reparameterisation. It buys nothing interpretable,
  and keeping them raw makes the fitted coefficients readable as per-unit
  effects.

The preprocessor is deliberately wrapped, never applied:

```python
Pipeline([("preprocess", build_preprocessor()), ("model", estimator)])
```

## 3. Leakage control

Imputation and scaling are pipeline steps, so any cross-validation or
resampling routine re-fits them on the training portion of each split by
construction. There is no code path where a caller can fit on the full frame.

The original notebook did this instead:

```python
scaler.fit(X)                                    # all 4,238 rows
scaled_data = scaler.transform(X)
X_train, X_test, ... = train_test_split(scaled_data, y, ...)
```

The scaler's parameters were a function of the test rows. `scripts/02_leakage_audit.py`
measures the gap — largest is totChol at 0.4661 mg/dL, or 0.0106 standard
deviations — and `tests/test_leakage.py` proves the corrected pipeline is immune
while also proving the leaky ordering genuinely changes the statistics (so the
first claim is not vacuous).

Honest framing: the leak's effect on a single 80/20 split is ~0.001 accuracy.
The methodological defect is real; it is not the reason the original's numbers
looked good. Reporting only accuracy is.

## 4. Models

Seven, each a complete pipeline:

| Model | Configuration | Why it is here |
|---|---|---|
| DummyClassifier | `strategy="prior"` | the baseline accuracy must beat |
| LogisticRegression | `max_iter=2000` | the original's model, correctly preprocessed |
| LogisticRegression | `class_weight="balanced"` | isolates the effect of imbalance weighting |
| SVC(kernel="linear") | `class_weight="balanced"`, Platt-scaled | the original's second model, imbalance-aware |
| RandomForestClassifier | 500 trees, `min_samples_leaf=5`, `balanced_subsample` | non-linear, low-variance baseline |
| HistGradientBoostingClassifier | `lr=0.05`, `max_iter=400`, early stopping | strong tabular learner |
| MLPClassifier | `hidden_layer_sizes=(100,100,100)`, `solver="adam"` | the original's third model, solver fixed |

The MLP keeps the original's architecture. The optimiser is changed from
`solver="sgd"` with `learning_rate_init=0.01` to Adam, because stochastic
gradient descent at a fixed 0.01 step on unscaled inputs is why the original's
MLP scored *below* its own logistic regression.

`SVC(probability=True)` was deprecated in scikit-learn 1.9, so the balanced
linear SVM is wrapped in `CalibratedClassifierCV(method="sigmoid", cv=3)` — the
supported equivalent, and one that produces genuinely calibrated probabilities
(mean predicted 0.1518 against an observed 0.1520).

All random seeds are 42.

## 5. Evaluation

**Cross-validation.** `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`
over all 4,238 rows. Means and standard deviations are reported together;
`scripts/03_cross_validation.py` implements the fold loop explicitly rather than
using `cross_validate` so the per-fold numbers are available.

**Holdout.** One stratified 80/20 split, kept only so the comparison with the
original is like-for-like (3,390 train / 848 test, 15.19% / 15.21% positive).

**Metrics.** accuracy, precision, recall, F1, ROC-AUC, PR-AUC (average
precision), Brier score. Accuracy is computed because a reviewer will ask for it,
and treated as the least informative column.

**Why PR-AUC carries the weight.** With 15.2% positives, a constant predictor
already scores PR-AUC 0.152. Precision–recall curves are sensitive to the
positive class in a way ROC curves are not at this prevalence, so PR-AUC is the
primary ranking metric.

**Why the Brier score is reported.** A model can rank well and still emit
probabilities that are wrong on the risk scale. The Brier score is the cheapest
way to catch that, and it caught `class_weight="balanced"` doubling it from
0.117 to 0.210.

**Thresholds.** Chosen on out-of-fold probabilities — every row scored by a model
that never saw it — so the selection is legitimate. `scripts/05` reports both the
F1-maximising threshold and the least selective threshold that still reaches 70%
sensitivity.

## 6. Calibration

`calibration_summary` reports calibration-in-the-large (`mean_predicted` against
the observed rate), the Brier score, a quantile-binned expected calibration
error, and the Spearman correlation of each model's ranking against the plain
logistic regression. Quantile bins are used so each bin holds an equal number of
participants and a few extreme predictions cannot dominate the statistic.

That table is the single most informative artifact in the repository: it is what
shows that `class_weight="balanced"` left the ranking essentially untouched
(ρ = 0.998) while multiplying everyone's predicted risk by 2.91.

## 7. Reproducibility

`python -m src --check` validates the data contract; `scripts/06_manifest.py`
writes the raw-file hashes and library versions. Each script mirrors its stdout
to `results/transcripts/`, so the committed CSVs and the committed logs come from
the same run. `tests/test_reference_numbers.py` asserts the numbers quoted in the
README against those CSVs, which makes stale prose a test failure rather than a
latent embarrassment.

## Known limitations

- **No external validation.** Every number is internal to one 4,238-participant
  cohort sampled from a single town in the mid-20th century. The true
  out-of-population performance of a Framingham-derived model is lower, and this
  holds especially for groups underrepresented in the original study.
- **Median imputation is a compromise**, not a solution, for the 388 missing
  glucose values. It shrinks imputed participants toward the centre of the
  distribution, which biases predictions toward the majority class.
- **The 5-fold standard deviations understate the uncertainty** in PR-AUC at 644
  events. A participant-level bootstrap would be required to make an interval
  claim.
- **The decision threshold is not resolved.** The analysis quantifies the
  trade-off; it does not choose. That choice depends on the relative cost of a
  missed event and a false alarm, which is a clinical and ethical question, not a
  statistical one.
- **`education` is treated as an unordered category** for imputation but enters
  the model as an integer. A proper treatment would use a monotone
  ordinal encoding or dummies.
