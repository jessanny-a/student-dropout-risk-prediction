from src.config import FINAL_THRESHOLD, MODEL_B_FEATURES, EXPECTED_FINAL_METRICS


def test_frozen_threshold():
    assert FINAL_THRESHOLD == 0.48


def test_model_b_feature_count():
    assert len(MODEL_B_FEATURES) == 37


def test_frozen_holdout_counts():
    assert EXPECTED_FINAL_METRICS["false_negatives"] == 36
    assert EXPECTED_FINAL_METRICS["false_positives"] == 41
