.PHONY: install check test eda audit cv holdout thresholds manifest notebook all clean

PY ?= python

install:
	$(PY) -m pip install -r requirements.txt

check:
	$(PY) -m src --check

test:
	$(PY) -m pytest tests/ -q

eda:
	$(PY) scripts/01_eda.py

audit:
	$(PY) scripts/02_leakage_audit.py

cv:
	$(PY) scripts/03_cross_validation.py

holdout:
	$(PY) scripts/04_holdout_evaluation.py

thresholds:
	$(PY) scripts/05_threshold_analysis.py

manifest:
	$(PY) scripts/06_manifest.py

# Builds and executes the notebook so the committed file carries its outputs.
notebook:
	$(PY) scripts/07_notebook.py

# Scripts run before tests so the reference-number tests have artifacts to check.
all: check eda audit cv holdout thresholds manifest notebook test

clean:
	rm -rf results/figures/*.png results/*.csv results/manifest.json
	rm -rf .pytest_cache .ipynb_checkpoints
	find . -name '__pycache__' -type d -prune -exec rm -rf {} +
