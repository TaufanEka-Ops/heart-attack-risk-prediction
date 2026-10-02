# Results

Everything here is generated. `make all` produces these files; nothing is
hand-edited. Each script also mirrors its console output to `transcripts/`, so
each CSV can be traced back to the run that wrote it.

## Metrics

| File | Written by | Contents |
|---|---|---|
| `metrics_cv.csv` | `scripts/03_cross_validation.py` | 7 models × 7 metrics, 5-fold CV mean and std |
| `metrics_holdout.csv` | `scripts/04_holdout_evaluation.py` | the same models on one stratified 80/20 split |
| `oof_probabilities.csv` | `scripts/04_holdout_evaluation.py` | out-of-fold predicted probabilities, one row per participant per model |
| `calibration_summary.csv` | `scripts/04_holdout_evaluation.py` | calibration-in-the-large, Brier, ECE, Spearman vs the plain logistic regression |
| `confusion_matrices.csv` | `scripts/04_holdout_evaluation.py` | tp/fp/fn/tn and derived scores at threshold 0.5 |
| `threshold_analysis.csv` | `scripts/05_threshold_analysis.py` | precision, recall, F1 and flagged share across thresholds 0.05–0.95 |
| `thresholds_for_target_recall.csv` | `scripts/05_threshold_analysis.py` | the operating point that reaches 70% sensitivity |
| `missingness.csv` | `scripts/01_eda.py` | missing counts and percentages |

## Audit of the original notebook

| File | Contents |
|---|---|
| `leakage_audit_imputation.csv` | imputation rule per column; 158 cells filled outside their own domain |
| `leakage_audit_scaler.csv` | scaler statistics on all rows vs the training split only |
| `leakage_audit_metric_impact.csv` | the same model run leaky and correctly, side by side |

## Figures

| File | Contents |
|---|---|
| `01_class_balance.png` | class balance with the majority-class share annotated |
| `02_missingness.png` | missing percentage per column |
| `03_correlation.png` | feature correlation heatmap including the target |
| `04_roc_curves.png` | ROC curves on out-of-fold probabilities, all models |
| `05_pr_curves.png` | precision–recall curves with the no-skill line |
| `06_confusion_matrix.png` | confusion matrix, best model at threshold 0.5 |
| `07_calibration.png` | calibration curves, out-of-fold |
| `08_model_comparison.png` | PR-AUC per model, mean ± std across folds |
| `08b_model_comparison_roc.png` | the same for ROC-AUC |
| `09_thresholds.png` | precision / recall / F1 against the decision threshold |

## Provenance

`manifest.json` records the raw CSV's SHA-256 and MD5, the row and positive
counts, the evaluation protocol, the library versions, and an inventory of every
artifact in this directory. `tests/test_reference_numbers.py` asserts the headline
numbers in `README.md` against these CSVs, so stale prose fails the test suite.
