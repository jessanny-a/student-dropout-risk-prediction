from pathlib import Path

RANDOM_STATE = 42
PREDICTION_HORIZON = "End of Semester 1"
FINAL_MODEL_NAME = "XGBoost"
FINAL_POLICY = "Balanced"
FINAL_THRESHOLD = 0.48

MIN_GROUP_SIZE = 30
MIN_POSITIVE_CASES = 10
MIN_NEGATIVE_CASES = 10

DATA_RAW = Path("data/raw")
DATA_PROCESSED = Path("data/processed")
MODELS = Path("models")
REPORTS = Path("reports")
FIGURES = REPORTS / "figures"

CATEGORICAL_FEATURES = [
    "marital_status", "application_mode", "course", "previous_qualification",
    "nationality", "mothers_qualification", "fathers_qualification",
    "mothers_occupation", "fathers_occupation",
]

BINARY_FEATURES = [
    "daytime_evening_attendance", "displaced", "educational_special_needs",
    "debtor", "tuition_fees_up_to_date", "gender", "scholarship_holder",
    "international",
]

ENROLLMENT_NUMERIC_FEATURES = [
    "application_order", "previous_qualification_grade", "admission_grade",
    "age_at_enrollment", "unemployment_rate", "inflation_rate", "gdp",
]

SEMESTER1_FEATURES = [
    "curricular_units_1st_sem_credited",
    "curricular_units_1st_sem_enrolled",
    "curricular_units_1st_sem_evaluations",
    "curricular_units_1st_sem_approved",
    "curricular_units_1st_sem_grade",
    "curricular_units_1st_sem_without_evaluations",
]

SEMESTER2_FEATURES = [
    "curricular_units_2nd_sem_credited",
    "curricular_units_2nd_sem_enrolled",
    "curricular_units_2nd_sem_evaluations",
    "curricular_units_2nd_sem_approved",
    "curricular_units_2nd_sem_grade",
    "curricular_units_2nd_sem_without_evaluations",
]

ENROLLMENT_ENGINEERED = [
    "academic_preparedness_gap",
    "academic_preparedness_mean",
    "financial_risk_count",
]

SEMESTER1_ENGINEERED = [
    "sem1_completion_rate",
    "sem1_evaluation_success_rate",
    "sem1_without_evaluation_rate",
    "sem1_credit_rate",
]

MODEL_A_FEATURES = (
    CATEGORICAL_FEATURES + BINARY_FEATURES
    + ENROLLMENT_NUMERIC_FEATURES + ENROLLMENT_ENGINEERED
)

MODEL_B_FEATURES = MODEL_A_FEATURES + SEMESTER1_FEATURES + SEMESTER1_ENGINEERED

MODEL_A_CATEGORICAL = CATEGORICAL_FEATURES
MODEL_A_NUMERIC = BINARY_FEATURES + ENROLLMENT_NUMERIC_FEATURES + ENROLLMENT_ENGINEERED

MODEL_B_CATEGORICAL = CATEGORICAL_FEATURES
MODEL_B_NUMERIC = (
    BINARY_FEATURES + ENROLLMENT_NUMERIC_FEATURES + ENROLLMENT_ENGINEERED
    + SEMESTER1_FEATURES + SEMESTER1_ENGINEERED
)

FROZEN_XGB_PARAMS = {
    "subsample": 1.0,
    "reg_lambda": 1,
    "reg_alpha": 0.1,
    "n_estimators": 300,
    "min_child_weight": 1,
    "max_depth": 3,
    "learning_rate": 0.05,
    "gamma": 0,
    "colsample_bytree": 1.0,
    "eval_metric": "logloss",
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}

EXPECTED_FINAL_METRICS = {
    "accuracy": 0.8939,
    "balanced_accuracy": 0.8902,
    "precision": 0.8581,
    "recall": 0.8732,
    "f1": 0.8656,
    "f2": 0.8702,
    "specificity": 0.9072,
    "roc_auc": 0.9540,
    "pr_auc": 0.9492,
    "alert_rate": 0.3981,
    "false_negatives": 36,
    "false_positives": 41,
}
