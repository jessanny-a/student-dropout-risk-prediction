# Student Dropout Risk Prediction

An explainable and fairness-audited machine-learning capstone project for identifying higher-education students who may benefit from proactive support at the **end of Semester 1**.

**Project status: Final capstone project**

## Final Model

- Model: **XGBoost**
- Prediction horizon: **End of Semester 1**
- Frozen operating threshold: **0.48**
- Training records: **2,904**
- Untouched holdout records: **726**
- ROC-AUC: **0.9540**
- PR-AUC: **0.9492**
- Balanced accuracy: **0.8902**
- Dropout recall: **0.8732**
- Precision: **0.8581**
- Deployment conclusion: **CONDITIONAL GO — SUPPORTIVE DECISION SUPPORT ONLY**

## Final Deliverables

- [Executed capstone notebook](notebooks/Student_Dropout_Capstone.ipynb)
- [Clean reproducibility notebook](notebooks/Student_Dropout_Capstone_No_Outputs.ipynb)
- [Technical Jupyter presentation](presentations/student_dropout_technical_jupyter_slides.ipynb)
- [Business PowerPoint presentation](presentations/student_dropout_business_deck.pptx)
- [Final report](reports/final_report.md)
- [Model card](docs/model_card.md)
- [Bias & Fairness Analysis](docs/bias_fairness_analysis.md)
- [Responsible AI and Governance](docs/responsible_ai.md)
- [Reproducibility guide](docs/reproducible_code.md)
- [Frozen XGBoost model artifact](models/final_student_dropout_xgboost.joblib)
- [Deployment configuration](models/final_deployment_config.json)

## Data

This project uses the **Predict Students' Dropout and Academic Success** dataset from the UCI Machine Learning Repository.

Realinho, V., Vieira Martins, M., Machado, J., & Baptista, L. (2021). *Predict Students' Dropout and Academic Success* [Dataset]. UCI Machine Learning Repository. DOI: `10.24432/C5MC89`.

The dataset is licensed under **CC BY 4.0** and is included at `data/raw/Predict Student Dropout.csv` for reproducibility. Dataset attribution and licensing are documented in `data/DATA_LICENSE_AND_ATTRIBUTION.md`.

## Repository Structure

```text
student-dropout-risk-prediction/
├── README.md
├── LICENSE
├── requirements.txt
├── pyproject.toml
├── src/                         # reusable modeling and governance code
├── notebooks/
│   ├── Student_Dropout_Capstone.ipynb
│   └── Student_Dropout_Capstone_No_Outputs.ipynb
├── data/
│   ├── raw/Predict Student Dropout.csv
│   ├── DATA_LICENSE_AND_ATTRIBUTION.md
│   └── README.md
├── models/
│   ├── final_student_dropout_xgboost.joblib
│   ├── final_deployment_config.json
│   └── README.md
├── reports/
│   ├── final_report.md
│   └── tables/
├── presentations/
│   ├── student_dropout_technical_jupyter_slides.ipynb
│   ├── student_dropout_business_deck.pptx
│   └── README.md
├── docs/
│   ├── model_card.md
│   ├── bias_fairness_analysis.md
│   ├── responsible_ai.md
│   ├── reproducibility.md
│   └── reproducible_code.md
└── tests/
```

## Reproduce the Frozen Final Model

Create a Python 3.13 environment, install the pinned dependencies, and run:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m src.run_pipeline --data "data/raw/Predict Student Dropout.csv"
```

Windows PowerShell activation:

```powershell
.venv\Scripts\Activate.ps1
```

The reproduction pipeline recreates source cleaning, resolved binary outcomes, feature engineering, the deterministic 80/20 split, the frozen XGBoost configuration, the 0.48 operating threshold, final holdout metrics, and primary fairness outputs.

## Methodology

The project covers:

1. problem framing and prediction-horizon design;
2. data validation and feature engineering;
3. baseline model comparison;
4. hyperparameter tuning with stratified cross-validation;
5. training-only threshold optimization;
6. untouched holdout evaluation;
7. SHAP, LIME, PDP, and ICE explainability;
8. subgroup fairness auditing;
9. demographic parity, disparate impact, equal opportunity, and equalised odds;
10. Responsible AI governance and production monitoring.

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

Detailed machine-readable results are available under `reports/tables/`.

## Explainability

Global SHAP analysis identified **Semester 1 completion rate** as the strongest predictor. SHAP probability reconstruction passed with a maximum difference of approximately `3.58e-07`.

PDP and ICE showed a strong nonlinear inverse association between Semester 1 completion and predicted dropout risk. LIME was used as a complementary local explanation method.

These techniques explain **model behavior and predictive associations, not causal effects**.

## Fairness

The largest supported equal-opportunity gaps were:

1. financial-risk group — **20.1 pp**
2. debtor status — **16.2 pp**
3. age — **15.1 pp**
4. scholarship status — **11.4 pp**

Formal fairness conclusions were restricted to groups meeting minimum support requirements. See `docs/bias_fairness_analysis.md` for the complete analysis.

## Responsible AI

The final deployment position is:

> **CONDITIONAL GO — SUPPORTIVE DECISION SUPPORT ONLY**

Required controls include human review, documented overrides, a non-alert safety net, recurring subgroup and calibration monitoring, program-level monitoring, privacy/access controls, and audit logging.

The model must not autonomously trigger withdrawal, dismissal, admission, scholarship denial, disciplinary action, or denial of institutional services.

## Evaluation Integrity

The 726-record final holdout was kept separate from model fitting, hyperparameter tuning, and threshold selection. Future mitigation, recalibration, or model redevelopment must use new development and validation data rather than tuning against the final holdout.

## License

Project source code is released under the **MIT License**. The source dataset remains separately licensed under **CC BY 4.0** by its original authors.
