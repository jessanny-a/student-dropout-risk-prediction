# Reproducibility Checklist

## Environment
- [ ] Record exact Python version.
- [ ] Pin exact package versions before publication.
- [ ] Preserve random state 42.

## Data
- [ ] Obtain the UCI dataset.
- [ ] Place it at `data/raw/Predict Student Dropout.csv`.
- [ ] Confirm 4,424 rows and 37 columns.
- [ ] Confirm no missing values and no exact duplicate rows.

## Evaluation Integrity
- [ ] Preserve the 726-record untouched final holdout.
- [ ] Verify zero train/holdout index overlap.
- [ ] Do not tune using the final holdout.
- [ ] Frozen threshold = 0.48.

## Expected Final Holdout Benchmark
- Accuracy: 0.8939
- Balanced accuracy: 0.8902
- Precision: 0.8581
- Recall: 0.8732
- F1: 0.8656
- F2: 0.8702
- Specificity: 0.9072
- ROC-AUC: 0.9540
- PR-AUC: 0.9492
- Alert rate: 0.3981
- False negatives: 36
- False positives: 41

## Responsible AI
**CONDITIONAL GO — SUPPORTIVE DECISION SUPPORT ONLY**
