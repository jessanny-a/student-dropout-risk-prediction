# Model Card — Student Dropout Risk Prediction

## Model Details
- Model family: XGBoost classifier
- Prediction target: Student dropout risk
- Prediction horizon: End of Semester 1
- Operating threshold: 0.48
- Training records: 2,904
- Final holdout records: 726

## Intended Use
Supportive, human-reviewed early-warning and outreach prioritization.

## Prohibited Use
The model must not autonomously trigger:
- withdrawal or dismissal;
- admission decisions;
- scholarship denial;
- disciplinary action;
- denial of institutional services.

## Final Holdout Performance
- Accuracy: 0.8939
- Balanced accuracy: 0.8902
- Precision: 0.8581
- Recall: 0.8732
- F1: 0.8656
- ROC-AUC: 0.9540
- PR-AUC: 0.9492
- False negatives: 36
- False positives: 41

## Explainability
Primary: SHAP. Complementary: LIME, PDP, ICE.

Semester 1 completion rate was the strongest global predictor.

## Fairness
Meaningful subgroup performance differences were observed, especially across financial-risk, debtor, age, and scholarship groups. Formal comparisons were limited to groups meeting predefined support rules.

## Governance
**CONDITIONAL GO — SUPPORTIVE DECISION SUPPORT ONLY**

Human oversight, alternative referral mechanisms, privacy controls, audit logging, calibration monitoring, and recurring subgroup monitoring are required.
