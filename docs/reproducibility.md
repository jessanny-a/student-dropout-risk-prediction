# Reproducibility

This document records the verified configuration for reproducing the frozen final model and benchmark.

## Environment

- Python requirement: **3.13**
- Random state: **42**
- Core package versions are pinned in `requirements.txt`.
- Project metadata and Python compatibility are defined in `pyproject.toml`.

## Data Contract

The source dataset is included at:

`data/raw/Predict Student Dropout.csv`

Expected source QA:

- 4,424 rows
- 37 total columns
- 36 predictors plus target
- 0 missing values
- 0 exact duplicate rows

The unresolved `Enrolled` class is excluded from the binary task, leaving 3,630 resolved records.

## Final Evaluation Split

- Training records: **2,904**
- Untouched holdout records: **726**
- Stratified 80/20 split
- Random state: **42**
- Verified train/holdout index overlap: **0**

The final holdout was not used for hyperparameter tuning or threshold optimization.

## Frozen Operating Configuration

- Model: **XGBoost**
- Prediction horizon: **End of Semester 1**
- Frozen threshold: **0.48**

## Expected Final Holdout Benchmark

| Metric | Expected result |
|---|---:|
| Accuracy | 0.8939 |
| Balanced accuracy | 0.8902 |
| Precision | 0.8581 |
| Recall | 0.8732 |
| F1 | 0.8656 |
| F2 | 0.8702 |
| Specificity | 0.9072 |
| ROC-AUC | 0.9540 |
| PR-AUC | 0.9492 |
| Alert rate | 0.3981 |
| False negatives | 36 |
| False positives | 41 |

## Reproduction Command

From the repository root:

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

## Responsible AI Constraint

The reproduced model remains subject to the final governance conclusion:

> **CONDITIONAL GO — SUPPORTIVE DECISION SUPPORT ONLY**

Future threshold changes, mitigation experiments, recalibration, or model redevelopment require new development and validation data.
