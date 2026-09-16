# Models

This directory contains the frozen final model artifact reproduced from the public UCI dataset.

- `final_student_dropout_xgboost.joblib` - preprocessing pipeline + final XGBoost classifier
- `final_deployment_config.json` - threshold, prediction horizon, data counts, and final metrics

Frozen operating threshold: **0.48**.

**Security note:** Joblib/pickle artifacts can execute code when loaded. Only load this file from a trusted copy of this repository and verify its SHA-256 hash against `MANIFEST_SHA256.txt`.
