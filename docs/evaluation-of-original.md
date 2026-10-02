# Evaluation of the original notebook

The source is a single Kaggle-style coursework notebook,
`heart-attack-analisys.ipynb`: load a CSV, mean-impute the missing values, scale
the whole frame, split 80/20, and fit three classifiers, reporting accuracy
only.

This document records what it got right, what it got wrong, and — because the
first two are easy to overstate — what the defects actually cost.

## What it reported

| Model | Reported accuracy |
|---|---|
| LogisticRegression(max_iter=2000) | 86.910% |
| `make_pipeline(MinMaxScaler(), SVC(kernel='linear'))` | 86.557% |
| MLPClassifier((100,100,100), solver='sgd', lr=0.01) | 85.613% |

## What it got right

- It did inspect the data before modelling: `data.info()` and
  `data.isnull().sum()` are both there, and the missing-value counts it printed
  are correct.
- It used `Pipeline` for the SVM, which is the correct instinct — a scaler needs
  to travel with the model.
- It compared three genuinely different model families rather than tuning one.

## Defect 1 — the imputation rule ignores column type

```python
data['education'].fillna(data['education'].mean(), inplace=True)
data['cigsPerDay'].fillna(data['cigsPerDay'].mean(), inplace=True)
data['BPMeds'].fillna(data['BPMeds'].mean(), inplace=True)
data['totChol'].fillna(data['totChol'].mean(), inplace=True)
data['BMI'].fillna(data['BMI'].mean(), inplace=True)
data['heartRate'].fillna(data['heartRate'].mean(), inplace=True)
data['glucose'].fillna(data['glucose'].mean(), inplace=True)
```

One rule — the column mean — applied to every column regardless of what the
column measures.

Consequences, measured by `scripts/02_leakage_audit.py`:

| Column | Type | Fill value | Legal domain | Cells out of domain |
|---|---|---|---|---|
| education | ordinal 1–4 | 1.9789 | {1,2,3,4} | 105 |
| BPMeds | binary 0/1 | 0.0296 | {0,1} | 53 |

**158 cells were assigned values that are not members of their own column's
domain.** A participant is recorded as having taken 0.03 of a blood-pressure
medication, and 105 participants have a fractional education level.

This is not a cosmetic problem. `BPMeds` and `education` are inputs to every
downstream model, and a fractional category is a value the model can never see
at prediction time from a real, complete record.

*Corrected by* giving each feature group its own imputation rule inside a
`ColumnTransformer`: median for continuous, most-frequent for the binary
indicators and for the ordinal education code.

## Defect 2 — preprocessing fitted before the split

```python
scaler = StandardScaler()
scaler.fit(X)                                   # all 4,238 rows
scaled_data = scaler.transform(X)
X_train, X_test, y_train, y_test = train_test_split(scaled_data, y, ...)
```

The imputation means and the scaler's location/scale parameters were computed
from the entire dataset, including the 848 rows that became the test set. Every
reported test metric is therefore contaminated: the test rows helped choose the
transformation used to score them.

Measured gap between the full-data and training-only statistics
(`results/leakage_audit_scaler.csv`):

| Column | All rows | Train only | Difference | In sd units |
|---|---|---|---|---|
| totChol | 236.7216 | 236.2554 | 0.4661 | 0.0106 |
| diaBP | 82.8935 | 82.8307 | 0.0628 | 0.0053 |
| cigsPerDay | 9.0031 | 9.0469 | 0.0438 | 0.0037 |
| heartRate | 75.8789 | 75.9203 | 0.0414 | 0.0034 |
| age | 49.5849 | 49.6260 | 0.0410 | 0.0048 |

*Corrected by* moving imputation and scaling inside the model pipeline, so
`cross_validate` and `train_test_split` re-fit them per fold. `tests/test_leakage.py`
proves the fitted statistics equal the training-only statistics, that swapping
the held-out frame for one 10,000× larger changes nothing about the fitted
model, and — so those proofs are not vacuous — that fitting on train+test
genuinely does move the statistics.

### The honest size of this defect

Running the same logistic regression both ways on the same split:

| Metric | Leaky preprocessing | Correct pipeline | Difference |
|---|---|---|---|
| accuracy | 0.8479 | 0.8491 | −0.0012 |
| precision | 0.5000 | 0.5294 | −0.0294 |
| recall | 0.0698 | 0.0698 | 0.0000 |
| F1 | 0.1224 | 0.1233 | −0.0008 |
| ROC-AUC | 0.6991 | 0.6996 | −0.0006 |
| PR-AUC | 0.2985 | 0.2972 | +0.0014 |
| Brier | 0.1221 | 0.1220 | +0.0001 |

**The leak moved nothing by more than 0.03, and in one case it moved the metric
the wrong way.** A scaler fitted on 20% extra rows produces almost the same mean
and variance.

This matters for how the original is criticised. Claiming "the leakage is why
the notebook's accuracy was high" would be false. The leakage is a genuine
methodological defect that must be fixed — it invalidates the reported numbers as
estimates of generalisation, and it grows with the number of features and the
complexity of the preprocessing — but it is a *small* leak with a *small*
effect, and saying otherwise would be its own kind of dishonesty.

## Defect 3 — accuracy as the only metric

This, not the leakage, is what makes the original's conclusion wrong.

The dataset is 15.2% positive. A classifier that answers "no CHD" for every
participant scores **accuracy 0.848** — the same 84.8% that appears in the
original's own results. The reported 86.910% is 2.1 percentage points above a
predictor that learns nothing.

What accuracy hides is visible the moment recall is computed. Cross-validated,
the same logistic regression on correctly preprocessed data gives:

| | Value |
|---|---|
| accuracy | 0.8551 |
| precision | 0.7099 |
| **recall** | **0.0807** |
| F1 | 0.1440 |
| ROC-AUC | 0.7250 |
| PR-AUC | 0.3466 |

**The model identifies 8.1% of the people who go on to develop coronary heart
disease.** At the default 0.5 threshold it flags 1.7% of the cohort, catches 52
of 644 events, and misses 592. Its high precision is simply the base rate
showing through: when you predict almost nothing positive, the few positives you
predict are likely to be real.

The original notebook also never reported a confusion matrix, never reported a
ROC or PR curve, never used cross-validation (a single `random_state=10` split,
so all three "comparisons" are one draw from a noisy distribution), and never
compared against a baseline.

## Defect 4 — the MLP was set up to fail

```python
MLPClassifier(hidden_layer_sizes=(100,100,100), max_iter=500,
              activation='relu', solver='sgd',
              learning_rate='constant', learning_rate_init=0.01)
```

Plain stochastic gradient descent at a fixed 0.01 step, no momentum, no adaptive
rate, no early stopping, on a 3,390-row training set. Unsurprisingly it
underperformed the linear model it was supposed to beat. That is a
hyperparameter problem, not evidence about neural networks. This repository keeps
the architecture and changes the optimiser to Adam with early stopping; the MLP
still does not beat logistic regression, but at least that result now means
something.

## Summary

| Issue | Severity | Fixed by |
|---|---|---|
| Mean-imputing categorical columns | Real data-integrity bug; 158 invalid cells | Per-group imputation rules in a `ColumnTransformer` |
| Preprocessing fitted before the split | Real methodological defect; ≤0.03 effect on this data | Pipeline-internal preprocessing; proven by tests |
| Accuracy-only reporting | **The headline error**; hid 8.1% recall | Accuracy, precision, recall, F1, ROC-AUC, PR-AUC, Brier, confusion matrix, majority baseline |
| Single train/test split | Results are one noisy draw | 5-fold stratified CV with means and standard deviations |
| MLP hyperparameters | Misleading "neural nets are worse" conclusion | Adam + early stopping, architecture retained |

The corrected pipeline produces **worse-looking** headlines than the original
(accuracy 0.8551 against 0.8691) and a far more useful picture: moderate
discrimination, calibrated probabilities, and a decision threshold that would
have to flag 41.8% of the cohort to reach 70% sensitivity.
