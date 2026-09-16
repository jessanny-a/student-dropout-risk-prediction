import pandas as pd

from src.data_loading import clean_column_name, build_binary_outcome
from src.feature_engineering import engineer_features

def test_clean_column_name():
    assert clean_column_name("Daytime/evening attendance\t") == "daytime_evening_attendance"
    assert clean_column_name("Mother's qualification") == "mothers_qualification"

def test_binary_outcome_excludes_enrolled():
    df = pd.DataFrame({"target": ["Dropout", "Graduate", "Enrolled"]})
    out = build_binary_outcome(df)
    assert out["dropout"].tolist() == [1, 0]
    assert "Enrolled" not in out["target"].tolist()

def test_feature_engineering_safe_zero_denominator():
    df = pd.DataFrame({
        "admission_grade": [120.0],
        "previous_qualification_grade": [110.0],
        "debtor": [1],
        "tuition_fees_up_to_date": [0],
        "curricular_units_1st_sem_approved": [0],
        "curricular_units_1st_sem_enrolled": [0],
        "curricular_units_1st_sem_evaluations": [0],
        "curricular_units_1st_sem_without_evaluations": [0],
        "curricular_units_1st_sem_credited": [0],
    })
    out = engineer_features(df)
    assert out.loc[0, "financial_risk_count"] == 2
    assert out.loc[0, "sem1_completion_rate"] == 0
