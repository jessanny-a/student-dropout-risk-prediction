import numpy as np
import pandas as pd

from src.evaluate import evaluate_threshold
from src.fairness_audit import formal_fairness_metrics

def test_threshold_metrics():
    y_true = np.array([0, 0, 1, 1])
    probs = np.array([0.1, 0.6, 0.4, 0.9])
    result = evaluate_threshold(y_true, probs, 0.5)
    assert result["false_positives"] == 1
    assert result["false_negatives"] == 1
    assert result["true_positives"] == 1
    assert result["true_negatives"] == 1

def test_formal_fairness_insufficient_support():
    table = pd.DataFrame({
        "sufficient_support": [True],
        "alert_rate": [0.5],
        "recall": [0.8],
        "false_positive_rate": [0.1],
    })
    result = formal_fairness_metrics({"one_group": table})
    assert result.loc[0, "assessment_status"] == "Insufficient comparable support"
