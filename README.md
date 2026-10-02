# Framingham 10-Year CHD Risk — a leakage-free, imbalance-aware rebuild

Ten-year coronary heart disease risk prediction on the Framingham Heart Study
subset: **4,238 participants, 16 columns, 644 CHD events (15.2% positive)**.

This repository rebuilds a Kaggle-style coursework notebook into something a
reviewer can audit. The modelling conclusions are unglamorous and that is the
point:

```
Best model          logistic regression, 5-fold CV
                    ROC-AUC 0.7250 ± 0.0201, PR-AUC 0.3466 ± 0.0341
No-skill baseline   PR-AUC 0.1520 (the prevalence itself)
Majority-class      accuracy 0.848 — i.e. predicting "no CHD" for everyone
Default threshold   at 0.5 the best model catches 0.081 of future cases
Threshold 0.20      recall 0.519, precision 0.308, F1 0.386
70% sensitivity     requires flagging 41.8% of the cohort; 3 of 4 flags are wrong
```

The honest summary: **these features rank risk moderately well and cannot
support a referral decision at the default threshold.** The original notebook
reported 86.9% accuracy and concluded the models worked; 84.8% of that is the
majority class.

---

## Problem statement

The Framingham Heart Study followed residents of Framingham, Massachusetts and
recorded cardiovascular risk factors alongside a ten-year outcome: whether the
participant developed coronary heart disease (`TenYearCHD`). The task is binary
classification of an imbalanced target — 15.2% of participants experienced an
event — using 15 demographic, behavioural and clinical features (age, sex,
smoking, blood pressure, cholesterol, BMI, glucose, diabetes and so on).

An earlier coursework notebook on this data achieved 86.9% accuracy with
logistic regression and stopped there. This repository asks the follow-up
questions: how much of that number is the base rate, how much of the training
pipeline leaked, and what operating characteristic would a clinic actually get?
The answers required fixing two preprocessing defects, replacing a single
train/test split with cross-validation, and reporting rank-based and calibrated
metrics alongside accuracy.

---

## Results

All numbers are 5-fold stratified cross-validated means ± standard deviations on
all 4,238 rows, produced by `scripts/03_cross_validation.py` and stored in
[`results/metrics_cv.csv`](results/metrics_cv.csv). The preprocessing pipeline is
re-fitted inside every fold, so no fold's validation rows influence its
imputation or scaling.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC | Brier |
|---|---|---|---|---|---|---|---|
| Dummy (majority class) | 0.848 ± 0.001 | 0.000 | 0.000 | 0.000 | 0.500 | 0.152 ± 0.001 | 0.129 |
| **Logistic Regression** | **0.8551 ± 0.0040** | **0.7099 ± 0.1240** | 0.0807 ± 0.0249 | 0.1440 ± 0.0403 | **0.7250 ± 0.0201** | **0.3466 ± 0.0341** | **0.1166** |
| Logistic Regression (balanced) | 0.6737 ± 0.0238 | 0.2701 ± 0.0191 | **0.6708 ± 0.0403** | **0.3849 ± 0.0234** | 0.7250 ± 0.0203 | 0.3446 ± 0.0342 | 0.2102 |
| Linear SVM (balanced) | 0.8530 ± 0.0024 | 0.7543 ± 0.1556 | 0.0497 ± 0.0141 | 0.0929 ± 0.0249 | 0.7245 ± 0.0175 | 0.3405 ± 0.0318 | 0.1171 |
| Random Forest (balanced) | 0.8188 ± 0.0111 | 0.3675 ± 0.0399 | 0.2608 ± 0.0337 | 0.3039 ± 0.0291 | 0.7098 ± 0.0258 | 0.3265 ± 0.0376 | 0.1413 |
| Hist Gradient Boosting | 0.8480 ± 0.0036 | 0.5281 ± 0.1506 | 0.0481 ± 0.0215 | 0.0866 ± 0.0347 | 0.6971 ± 0.0172 | 0.3151 ± 0.0400 | 0.1199 |
| MLP (100,100,100) | 0.8492 ± 0.0032 | 0.4629 ± 0.2780 | 0.0481 ± 0.0297 | 0.0862 ± 0.0522 | 0.7020 ± 0.0537 | 0.3009 ± 0.0519 | 0.1202 |

Reference: the no-skill PR-AUC is 0.1520, so the best model is **2.3× better than
guessing** on the metric that ignores the majority class. The single stratified
80/20 holdout (the protocol the original used) is in
[`results/metrics_holdout.csv`](results/metrics_holdout.csv) and agrees within
fold noise.

### The comparison is honest, so the gaps are small

Every model that is not a dummy sits between PR-AUC 0.30 and 0.35 — a spread
smaller than twice the fold standard deviation. Cross-validation says **the
choice of algorithm barely matters here**; a regularised linear model is as good
as a 500-tree forest or a 3-hidden-layer network. Reporting a single split, as
the original did, would have shown differences that are mostly noise.

### Accuracy was the wrong headline

Predicting "no CHD" for everyone scores **accuracy 0.848**. Logistic regression
scores 0.8551 — better by 0.7 of a percentage point, while catching only
**8.1% of the people who go on to develop heart disease** (recall 0.081, i.e. 52
of 644 events across the folds). On this dataset accuracy is a measure of how
imbalanced the data is, not of how well the model works. That is why the table
above leads with PR-AUC, ROC-AUC and Brier alongside accuracy.

### Discrimination is not clinical usefulness

ROC-AUC 0.7250 is real but modest, and it says nothing about what happens at a
decision point. Using the out-of-fold probabilities
([`results/threshold_analysis.csv`](results/threshold_analysis.csv)):

| Decision threshold | Precision | Recall | F1 | Share of cohort flagged |
|---|---|---|---|---|
| 0.50 (the silent default) | 0.703 | 0.081 | 0.145 | 1.7% |
| 0.30 | 0.352 | 0.256 | 0.296 | 11.1% |
| 0.20 (best F1) | 0.308 | 0.519 | 0.386 | 25.6% |
| 0.14 (70% sensitivity) | 0.257 | 0.706 | 0.377 | 41.8% |

To catch **70% of the 644 people who will develop CHD you must flag 1,770 of
4,238 people — 41.8% of the cohort — and 1,315 of those flags will be false
alarms.** A quarter of everyone screened would receive a positive result that is
wrong. This model is a **risk-ranking tool, not a referral rule**; the README
says so rather than letting a headline ROC-AUC imply otherwise.

### `class_weight="balanced"` buys an operating point, not accuracy

The balanced logistic regression reports recall 0.6708 against 0.0807 — a
factor of eight — while its ROC-AUC is **identical to four decimal places
(0.7250)**. The calibration table
([`results/calibration_summary.csv`](results/calibration_summary.csv)) explains
why:

| Model | Mean predicted risk | Observed risk | Ratio | Brier | Spearman vs plain LR |
|---|---|---|---|---|---|
| Logistic Regression | 0.1518 | 0.1520 | 0.999 | 0.117 | 1.000 |
| Logistic Regression (balanced) | 0.4422 | 0.1520 | **2.910** | 0.210 | **0.998** |
| Linear SVM (balanced, Platt) | 0.1518 | 0.1520 | 0.999 | 0.117 | 0.992 |
| Random Forest (balanced) | 0.2900 | 0.1520 | 1.909 | 0.141 | 0.903 |
| Hist Gradient Boosting | 0.1490 | 0.1520 | 0.980 | 0.120 | 0.835 |
| MLP (100,100,100) | 0.1514 | 0.1520 | 0.997 | 0.120 | 0.878 |

The balanced model **ranks people in almost exactly the same order**
(Spearman ρ = 0.998) but multiplies everyone's predicted probability by ~3. That
inflation is what makes the fixed 0.5 cut-off catch more cases. The same effect
is available for free — and reversibly, without destroying calibration — by
lowering the threshold on the well-calibrated model. The plain logistic
regression's mean predicted risk of 0.1518 against an observed 0.1520 is the
number that makes its probabilities usable as probabilities.

---

## What was wrong with the original notebook

Both defects are reproduced and measured, not just asserted, by
`scripts/02_leakage_audit.py`. Results in
[`results/leakage_audit_imputation.csv`](results/leakage_audit_imputation.csv) and
[`results/leakage_audit_scaler.csv`](results/leakage_audit_scaler.csv).

**Defect A — one imputation rule for every column type.** The original filled
every incomplete column with that column's mean:

```python
data['education'].fillna(data['education'].mean(), inplace=True)
data['BPMeds'].fillna(data['BPMeds'].mean(), inplace=True)
```

`education` is an ordinal code in {1,2,3,4} and `BPMeds` is a 0/1 indicator.
Their means are 1.9789 and 0.0296 — not categories. That put **158 cells** (105
education, 53 BPMeds) outside the domain of their own column. Fixed with a
`ColumnTransformer` that gives each feature group its own rule: median for
continuous, most-frequent for the binary indicators and for education.

**Defect B — preprocessing fitted before the split.** The original called
`scaler.fit(X)` on all 4,238 rows and only then ran `train_test_split`. The
scaler's means and variances were therefore functions of the rows that later
became the test set. Because the scaler saw ~20% extra data, the gap is small —
the largest is `totChol`, 236.7216 (all rows) against 236.2554 (train only),
0.4661 mg/dL or **0.0106 standard deviations** — and on a single split it moves
the test metrics only in the third decimal place (accuracy 0.8479 leaky versus
0.8491 correct).

**Read that honestly.** Scaling leakage is a real methodological defect — the
test rows helped choose the parameters used to score them, and it would inflate
more with a larger feature set or a different split — but it is *not* why the
original's numbers looked good. The reason the original analysis over-claimed is
that it never computed recall, never compared against a majority-class baseline,
and never used cross-validation. Fixing the leak is necessary; reporting the
right metrics is what changes the conclusion.

---

## Repository layout

```
data/raw/framingham.csv        the source data, unmodified (SHA-256 in results/manifest.json)
src/                           the library
  config.py                    paths, feature groups, protocol constants, data contract
  data.py                      loading and validation only — never imputes, never splits
  preprocess.py                ColumnTransformer + Pipeline (the leakage fix lives here)
  models.py                    seven complete pipelines, baselines included
  evaluate.py                  metrics, CV, holdout, calibration, thresholds
  plots.py                     every figure in results/figures
  leakage.py                   faithful reproduction of the original's defects, audit-only
  reporting.py                 transcript mirroring and CSV writing
  __main__.py                  `python -m src --check`
scripts/                       numbered, individually runnable experiments
  01_eda.py                    data contract, missingness, correlations, 3 figures
  02_leakage_audit.py          measures both defects
  03_cross_validation.py       the headline table
  04_holdout_evaluation.py     holdout + out-of-fold probabilities + calibration + 4 figures
  05_threshold_analysis.py     operating points and the 70%-sensitivity cost
  06_manifest.py               provenance manifest with the raw-file hash
tests/                         87 pytest tests
notebooks/                     01_eda_and_cv.ipynb — executed, outputs embedded
docs/                          methodology.md, evaluation-of-original.md
results/                       metrics CSVs, figures, per-script transcripts, manifest
```

The one executable notebook is committed **already executed**, so its outputs are
in the file. It is regenerated by `make notebook`, which runs it with `nbclient`.

---

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

make check        # validate the data contract (columns, ranges, missing cells)
make test         # 87 pytest tests, including the leakage proofs
make all          # every experiment, in order, writing results/
```

Individual targets: `make eda`, `make audit`, `make cv`, `make holdout`,
`make thresholds`, `make manifest`, `make notebook`, `make clean`.

`make all` runs the following, each writing a transcript to
`results/transcripts/`:

```bash
python -m src --check
python scripts/01_eda.py
python scripts/02_leakage_audit.py
python scripts/03_cross_validation.py
python scripts/04_holdout_evaluation.py
python scripts/05_threshold_analysis.py
python scripts/06_manifest.py
python -m pytest tests/ -q
```

Runtime is roughly 3 minutes on this machine (measured 3m18s for a full
`make clean && make all`); the slowest steps are the 5-fold cross-validation of
seven models and the notebook execution.

---

## Method notes

**Leakage is prevented structurally, not by discipline.** Imputation and scaling
are steps *inside* an `sklearn.Pipeline`, so `cross_validate` re-fits them per
fold automatically. There is no code path in which a caller can fit the
preprocessor on the full frame and split afterwards. `tests/test_leakage.py`
proves three separate things: the fitted statistics equal the training-only
statistics; swapping the held-out set for values 10,000× larger does not change
a single fitted parameter or prediction; and — so the first two are not vacuous —
fitting on train+test genuinely does move the statistics.

**Imputation is per feature group.** Median for the eight continuous features,
most-frequent for the six binary indicators and for the four-level education
code. Continuous features are standardised; binary indicators are left as 0/1
because scaling them is a linear reparameterisation with no interpretability
payoff.

**A baseline is mandatory.** The majority-class dummy is in the model table on
purpose. Any comparison that omits it on a 15%-positive target is not a
comparison.

**Cross-validation, not one split.** 5-fold stratified, `random_state=42`; means
and standard deviations are reported together. The 80/20 holdout is kept only to
match the original's protocol like-for-like.

**Thresholds are chosen on out-of-fold probabilities.** Every row's probability
comes from a model that never saw that row, so selecting an operating point from
these numbers is legitimate. Choosing a threshold on in-sample predictions, or
on the test set, would be another form of leakage.

**Calibration is reported alongside discrimination.** A model can rank well and
still produce probabilities that are unusable, as the balanced logistic
regression demonstrates (calibration ratio 2.91).

---

## What I would do differently

1. **Model time-to-event, not a binary flag.** `TenYearCHD` collapses ten years
   of follow-up into one label, so a person who has an event in year 1 and one
   who has it in year 10 count the same. Cox proportional hazards or a discrete-time
   survival model would use the follow-up structure that is already implied by
   the study design and would produce a risk *at a given horizon*, which is what
   a clinician actually asks for.

2. **Do not impute `glucose` with a median.** 388 of 4,238 glucose values are
   missing (9.2%), the largest single gap. Missingness is not obviously random —
   participants missing BMI show a 52.6% CHD rate against 15.0% for the rest
   (n = 19, so treat that figure with suspicion) — and median imputation shrinks
   every imputed person toward the middle of the distribution, which biases the
   model toward the majority class. Multiple imputation by chained equations, or
   an explicit missing-indicator column, would be more defensible.

3. **Report confidence intervals on the metrics, not just fold standard
   deviations.** Fold-to-fold variability understates the uncertainty in
   PR-AUC at 644 events. A bootstrap over participants would give intervals that
   support a claim like "PR-AUC is between X and Y", which is what a decision
   about whether to use the model needs.

4. **Test whether the model is worth more than a 3-variable score.** Age, sex
   and systolic blood pressure dominate the correlation table (age r = 0.225,
   sysBP r = 0.216). A parsimonious three-feature model would likely match these
   numbers, and demonstrating that would say more than adding a fourth algorithm.

5. **Decide the operating point with a stated cost ratio.** The threshold
   analysis shows the trade-off but does not resolve it. Choosing 0.14 versus
   0.20 is a clinical and ethical decision that depends on the harm of a missed
   event relative to a false alarm, and it should be stated, justified, and
   revisited as those costs change — not silently left at 0.5.

---

## Licence and data

Code: MIT — see [`LICENSE`](LICENSE).

Data: the Framingham Heart Study subset is a public teaching dataset derived
from NHLBI-funded research. It is included here for reproducibility; check the
upstream terms before redistributing it. The raw file's SHA-256 and MD5 are
recorded in [`results/manifest.json`](results/manifest.json) so a reviewer can
confirm they are looking at the same bytes.

**This is a statistics portfolio project, not a medical device.** Nothing here
should be used to make decisions about any person's health.
