# Reproducible Code Guide

The final notebook has been refactored into reusable modules under `src/`.

## Module Map

- `config.py` — feature groups, support rules, frozen threshold, final XGBoost parameters
- `data_loading.py` — source loading, column cleaning, resolved binary target
- `feature_engineering.py` — academic preparedness and Semester 1 engineered features
- `preprocessing.py` — imputation, one-hot encoding, numeric scaling
- `train.py` — baseline pipelines, tuning search constructors, frozen final model
- `evaluate.py` — deterministic split, threshold evaluation, holdout metrics
- `explainability.py` — SHAP helpers, raw-feature aggregation, PDP/ICE
- `fairness_audit.py` — subgroup metrics, support rules, disparity gaps, formal fairness metrics
- `governance.py` — permitted/prohibited use, guardrails, monitoring, overrides
- `run_pipeline.py` — end-to-end frozen-model reproduction

## Quick Reproduction

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m src.run_pipeline --data "data/raw/Predict Student Dropout.csv"
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

The reproduction command recreates the source cleaning, binary outcome, feature engineering, deterministic 80/20 split, frozen XGBoost configuration, 0.48 threshold, holdout evaluation, and primary fairness outputs.

## Full Search Reproduction

The notebook contains the full Logistic Regression, Random Forest, and XGBoost hyperparameter searches. Matching search constructors are available in `src/train.py`.

The practical reproduction path uses the final XGBoost parameters recorded by the training-only randomized search.

## Evaluation Integrity

Do not tune parameters or the threshold using the final 726-record holdout. Future mitigation experiments require new development and validation data.
