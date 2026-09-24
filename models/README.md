# Models

This directory contains the frozen final model artifact and its deployment metadata.

- `final_student_dropout_xgboost.joblib` — preprocessing pipeline plus final XGBoost classifier
- `final_deployment_config.json` — prediction horizon, frozen threshold, source-data QA, record counts, and final holdout metrics

## Frozen Configuration

- Model: **XGBoost**
- Prediction horizon: **End of Semester 1**
- Operating threshold: **0.48**
- Approved use: **Supportive decision support only**

## Security Note

Joblib/pickle artifacts can execute code when deserialized. Only load the model artifact from a trusted copy of this repository and in a controlled Python environment.

The model artifact is provided for reproducibility and demonstration. Operational deployment should also enforce the governance controls documented in `docs/responsible_ai.md`.
