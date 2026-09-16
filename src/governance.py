import pandas as pd

OVERRIDE_REASON_CATEGORIES = [
    "Additional academic information",
    "Additional financial information",
    "Student self-referral",
    "Faculty or advisor referral",
    "Recent circumstances not captured by model",
    "Data quality concern",
    "Existing support already in place",
    "Alert judged not actionable",
    "Other documented reason",
]

def model_use_governance():
    return pd.DataFrame([
        {"use_case": "Prioritize supportive student outreach", "status": "PERMITTED", "condition": "Human review and supportive intervention only."},
        {"use_case": "Offer academic advising or counseling", "status": "PERMITTED", "condition": "Participation and outreach should avoid punitive framing."},
        {"use_case": "Offer financial-support information", "status": "PERMITTED", "condition": "Risk indicators may be used to offer assistance, not to restrict access."},
        {"use_case": "Human case prioritization", "status": "PERMITTED WITH CONTROLS", "condition": "Risk score must be considered alongside other student information and professional judgment."},
        {"use_case": "Automatic withdrawal or dismissal decision", "status": "PROHIBITED", "condition": "The model is not designed for autonomous adverse decisions."},
        {"use_case": "Admission decision", "status": "PROHIBITED", "condition": "The model was not developed or validated for admission use."},
        {"use_case": "Scholarship denial", "status": "PROHIBITED", "condition": "Scholarship status contributes to predictions and subgroup disparities were observed."},
        {"use_case": "Disciplinary action", "status": "PROHIBITED", "condition": "Dropout-risk prediction is unrelated to disciplinary intent."},
        {"use_case": "Automatic denial of institutional services", "status": "PROHIBITED", "condition": "The score should expand support, not restrict opportunity."},
    ])

def deployment_guardrails():
    return pd.DataFrame([
        {"control": "Human oversight", "requirement": "A trained staff member reviews alerts before any consequential action."},
        {"control": "Frozen operating policy", "requirement": "Production model uses XGBoost with threshold 0.48 until a formally validated replacement is approved."},
        {"control": "No adverse autonomous action", "requirement": "A model prediction alone cannot trigger dismissal, discipline, denial, or loss of eligibility."},
        {"control": "Non-alert safety net", "requirement": "Students not flagged by the model remain eligible for support through other referral channels."},
        {"control": "Explainability availability", "requirement": "SHAP-based explanations are available for model review and quality assurance."},
        {"control": "Subgroup monitoring", "requirement": "Recall, FNR, FPR, precision, alert rate, and calibration are monitored across defined subgroups."},
        {"control": "Minimum subgroup support", "requirement": "Fairness conclusions are not drawn from groups with insufficient positive or negative outcome support."},
        {"control": "Program monitoring", "requirement": "Course-level performance is reviewed where sufficient sample support exists."},
        {"control": "Privacy control", "requirement": "Individual risk scores are visible only to authorized personnel with a legitimate support function."},
        {"control": "Audit logging", "requirement": "Model version, prediction time, threshold, and intervention decision should be recorded for later audit."},
        {"control": "Override documentation", "requirement": "Authorized overrides must record decision, reason, reviewer role, and follow-up action."},
    ])

def production_monitoring_plan():
    return pd.DataFrame([
        {"monitoring_area": "Data quality", "metric": "Schema, missingness, category changes, invalid values", "frequency": "Every scoring cycle", "action_if_concerning": "Stop scoring if required inputs are invalid or preprocessing assumptions are violated."},
        {"monitoring_area": "Prediction distribution", "metric": "Mean risk score and score distribution", "frequency": "Every scoring cycle", "action_if_concerning": "Investigate material distribution shift relative to baseline."},
        {"monitoring_area": "Operational workload", "metric": "Alert rate", "frequency": "Every scoring cycle", "action_if_concerning": "Review whether workload changes reflect population shift or model drift."},
        {"monitoring_area": "Overall performance", "metric": "Recall, precision, F1, balanced accuracy, ROC-AUC, PR-AUC", "frequency": "When outcomes mature", "action_if_concerning": "Investigate drift and consider formal model review."},
        {"monitoring_area": "Fairness", "metric": "Subgroup recall, FNR, FPR, precision, alert rate", "frequency": "Each completed cohort", "action_if_concerning": "Investigate subgroup degradation and determine whether mitigation or model redevelopment is needed."},
        {"monitoring_area": "Calibration", "metric": "Observed dropout rate versus mean predicted risk", "frequency": "Each completed cohort", "action_if_concerning": "Review probability calibration using development or validation data, not the final test set."},
        {"monitoring_area": "Program robustness", "metric": "Course-level recall, FPR, precision, balanced accuracy", "frequency": "Each completed cohort", "action_if_concerning": "Investigate persistently under-performing programs."},
        {"monitoring_area": "False-negative review", "metric": "Representative missed-dropout cases", "frequency": "Each completed cohort", "action_if_concerning": "Identify missing signals or new risk pathways for future model development."},
        {"monitoring_area": "Human override behavior", "metric": "Override rate, override direction, reason category, and follow-up action", "frequency": "Quarterly and each completed cohort", "action_if_concerning": "Investigate unusually high override rates or systematic staff/model disagreement."},
    ])

def empty_override_log():
    return pd.DataFrame(columns=[
        "case_id", "prediction_timestamp", "model_version", "decision_threshold",
        "predicted_dropout_probability", "model_alert", "human_final_decision",
        "override_type", "override_reason_category", "override_reason_notes",
        "reviewer_role", "follow_up_action", "review_timestamp",
    ])

def record_override(override_log, case_id, prediction_timestamp, model_version,
                    decision_threshold, predicted_dropout_probability,
                    model_alert, human_final_decision, override_reason_category,
                    override_reason_notes, reviewer_role, follow_up_action,
                    review_timestamp):
    if override_reason_category not in OVERRIDE_REASON_CATEGORIES:
        raise ValueError("override_reason_category is not approved.")

    if model_alert == 1 and human_final_decision == 0:
        override_type = "Model alert not actioned"
    elif model_alert == 0 and human_final_decision == 1:
        override_type = "Human-initiated intervention"
    else:
        override_type = "No override"

    new_record = pd.DataFrame([{
        "case_id": case_id,
        "prediction_timestamp": prediction_timestamp,
        "model_version": model_version,
        "decision_threshold": decision_threshold,
        "predicted_dropout_probability": predicted_dropout_probability,
        "model_alert": model_alert,
        "human_final_decision": human_final_decision,
        "override_type": override_type,
        "override_reason_category": override_reason_category,
        "override_reason_notes": override_reason_notes,
        "reviewer_role": reviewer_role,
        "follow_up_action": follow_up_action,
        "review_timestamp": review_timestamp,
    }])
    return pd.concat([override_log, new_record], ignore_index=True)
