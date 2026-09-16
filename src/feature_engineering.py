import numpy as np
import pandas as pd

def safe_divide(numerator, denominator):
    return np.where(denominator > 0, numerator / denominator, 0)

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["academic_preparedness_gap"] = (
        out["admission_grade"] - out["previous_qualification_grade"]
    )
    out["academic_preparedness_mean"] = out[
        ["admission_grade", "previous_qualification_grade"]
    ].mean(axis=1)
    out["financial_risk_count"] = (
        out["debtor"] + (1 - out["tuition_fees_up_to_date"])
    )
    out["sem1_completion_rate"] = safe_divide(
        out["curricular_units_1st_sem_approved"],
        out["curricular_units_1st_sem_enrolled"],
    )
    out["sem1_evaluation_success_rate"] = safe_divide(
        out["curricular_units_1st_sem_approved"],
        out["curricular_units_1st_sem_evaluations"],
    )
    out["sem1_without_evaluation_rate"] = safe_divide(
        out["curricular_units_1st_sem_without_evaluations"],
        out["curricular_units_1st_sem_enrolled"],
    )
    out["sem1_credit_rate"] = safe_divide(
        out["curricular_units_1st_sem_credited"],
        out["curricular_units_1st_sem_enrolled"],
    )
    return out
