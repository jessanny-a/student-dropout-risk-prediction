# Student Dropout Risk Prediction

An explainable and fairness-audited machine-learning capstone project for identifying students who may benefit from proactive support at the **end of Semester 1**.

## Final Result

- Final model: **XGBoost**
- Frozen threshold: **0.48**
- Training records: **2,904**
- Untouched holdout: **726**
- ROC-AUC: **0.9540**
- PR-AUC: **0.9492**
- Balanced accuracy: **0.8902**
- Dropout recall: **0.8732**
- Precision: **0.8581**
- Deployment conclusion: **CONDITIONAL GO — SUPPORTIVE DECISION SUPPORT ONLY**

## Data

Source: Realinho, V., Vieira Martins, M., Machado, J., & Baptista, L. (2021). *Predict Students' Dropout and Academic Success*. UCI Machine Learning Repository. DOI: `10.24432/C5MC89`.

License: **CC BY 4.0**.

The raw CSV is not bundled by default. See `data/README.md`.

## Repository Structure

```text
student-dropout-risk-prediction/
├── README.md
├── requirements.txt
├── src/
│   ├── config.py
│   ├── data_loading.py
│   ├── feature_engineering.py
│   ├── preprocessing.py
│   ├── train.py
│   ├── evaluate.py
│   ├── explainability.py
│   ├── fairness_audit.py
│   ├── governance.py
│   └── run_pipeline.py
├── notebooks/
│   ├── 01_student_dropout_capstone.ipynb
│   └── 01_student_dropout_capstone_no_outputs.ipynb
├── data/
├── models/
├── reports/
│   └── final_report.md
├── presentations/
├── docs/
└── tests/
```

## Quick Start

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

## Methodology

The project covers problem framing, data validation, feature engineering, baseline comparison, hyperparameter tuning, training-only threshold optimization, untouched holdout evaluation, SHAP/LIME/PDP/ICE explainability, subgroup fairness auditing, formal fairness metrics, and Responsible AI governance.

## Final Holdout Metrics

| Metric | Result |
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

## Explainability

Global SHAP identified **Semester 1 completion rate** as the strongest predictor. SHAP probability reconstruction passed with a maximum difference of approximately `3.58e-07`.

PDP and ICE confirmed a strong nonlinear inverse association between Semester 1 completion and predicted dropout risk. LIME was used as a complementary local explanation method.

Explanations are descriptions of **model behavior, not causality**.

## Fairness

Largest equal-opportunity gaps:

1. financial-risk group — **20.1 pp**
2. debtor status — **16.2 pp**
3. age — **15.1 pp**
4. scholarship status — **11.4 pp**

See `docs/bias_fairness_analysis.md`.

## Responsible AI

The model must be used for supportive decision support only. Required controls include human review, override documentation, a non-alert safety net, recurring fairness/calibration/program monitoring, privacy controls, and audit logging.

## Reproducibility

See:

- `docs/reproducible_code.md`
- `docs/reproducibility.md`
- `docs/notebook_review.md`

The final 726-record holdout must remain untouched by future tuning or mitigation work.

## Presentations

- `presentations/student_dropout_technical_deck.pptx`
- `presentations/student_dropout_business_deck.pptx`

## Final Report

See `reports/final_report.md`.

## Tests

```bash
pytest
```

## Publication Checklist

See `docs/github_publication_checklist.md`.

## License

Project source code is released under the **MIT License**. The source dataset remains separately licensed under **CC BY 4.0** by its original authors.
