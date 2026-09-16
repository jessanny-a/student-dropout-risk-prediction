# Final Report — Predicting Student Dropout Risk Using Explainable and Fair Machine Learning

## Executive Summary

This project develops an early-warning machine-learning system to identify higher-education students who may benefit from proactive support. The final model is an XGBoost classifier operating at the end of Semester 1 with a frozen probability threshold of 0.48.

On a 726-record untouched holdout set, the model achieved 0.8939 accuracy, 0.8902 balanced accuracy, 0.8581 precision, 0.8732 dropout recall, 0.8656 F1, 0.9540 ROC-AUC, and 0.9492 PR-AUC. The model produced 36 false negatives and 41 false positives.

The Responsible AI conclusion is **CONDITIONAL GO — SUPPORTIVE DECISION SUPPORT ONLY**.

## 1. Business Problem

The system is intended to help student-success teams prioritize supportive academic, financial, counseling, or advisory outreach. It is not intended to make autonomous adverse decisions.

## 2. Data

The project uses the UCI Machine Learning Repository's *Predict Students' Dropout and Academic Success* dataset (Realinho et al., 2021; DOI `10.24432/C5MC89`), licensed CC BY 4.0.

The source contains 4,424 records, 36 predictors, and the target classes Graduate, Dropout, and Enrolled. The unresolved Enrolled class was excluded from the primary binary task, leaving 3,630 resolved-outcome records.

## 3. Prediction Design

Two horizons were explored:

- Model A — enrollment-time risk
- Model B — end-of-first-semester risk

The final model uses Model B. Semester 2 variables were excluded to preserve the early-warning design.

## 4. Feature Engineering and Preprocessing

Engineered features include academic preparedness gap/mean, financial-risk count, Semester 1 completion rate, evaluation success rate, rate without evaluation, and credit rate.

Categorical features are most-frequent imputed and one-hot encoded. Numeric features are median-imputed and standardized. Preprocessing is fitted only on training data.

## 5. Model Development

Baseline models included Dummy, Logistic Regression, Decision Tree, Random Forest, and XGBoost. Logistic Regression, Random Forest, and XGBoost were tuned using five-fold stratified cross-validation.

Best tuned cross-validation ROC-AUC:

- Logistic Regression: 0.9330
- Random Forest: 0.9341
- XGBoost: 0.9373

XGBoost was selected as the final algorithm.

## 6. Threshold Optimization

Threshold optimization used training-only out-of-fold probabilities. The balanced XGBoost threshold of 0.48 maximized F1 and balanced accuracy in the training-only analysis while maintaining a practical alert workload.

The final threshold was frozen before holdout evaluation.

## 7. Final Holdout Evaluation

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

## 8. Explainability

SHAP, LIME, PDP, and ICE were used. Semester 1 completion rate was the dominant global SHAP feature. SHAP reconstruction validation produced a maximum probability difference of approximately `3.58e-07`.

PDP and ICE showed a strong nonlinear inverse association between Semester 1 completion and predicted dropout risk. These results describe model behavior rather than causal effects.

## 9. Bias and Fairness

Formal fairness metrics were calculated only for groups satisfying predefined minimum sample and class-support rules.

| Audit dimension | Demographic parity gap | Disparate impact ratio | Equal opportunity gap | Equalised odds gap |
|---|---:|---:|---:|---:|
| Financial risk | 51.1 pp | 0.357 | 20.1 pp | 20.1 pp |
| Debtor status | 42.5 pp | 0.451 | 16.2 pp | 16.2 pp |
| Age | 47.0 pp | 0.373 | 15.1 pp | 15.1 pp |
| Scholarship status | 33.5 pp | 0.300 | 11.4 pp | 11.4 pp |
| Gender | 23.2 pp | 0.568 | 3.6 pp | 6.0 pp |
| Displacement | 15.3 pp | 0.683 | 4.9 pp | 4.9 pp |
| Tuition status | Not comparable | Not comparable | Not comparable | Not comparable |

The most important monitoring priorities are financial-risk status, debtor status, age, and scholarship status.

## 10. Limitations

Limitations include moderate class imbalance, temporal leakage risk if Semester 1 data are used prematurely, residual overfitting risk, 36 false negatives, subgroup disparities, small subgroup support in some comparisons, reliance on observable institutional signals, the non-causal nature of explainability techniques, LIME transformed-feature interpretation, and external-validity/model-drift risk.

## 11. Responsible AI and Governance

Required controls include human review, documented overrides, a non-alert safety net, recurring subgroup and program monitoring, calibration monitoring, privacy/access controls, and audit logging.

The model must not autonomously trigger withdrawal, dismissal, admission, scholarship denial, disciplinary action, or denial of institutional services.

## 12. Conclusion

The final XGBoost model provides strong predictive discrimination and useful early-warning capability, but performance is heterogeneous across subgroups and false negatives remain.

> **CONDITIONAL GO — SUPPORTIVE DECISION SUPPORT ONLY**

Future mitigation, recalibration, or threshold changes must use new development and validation data.
