import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

def stratified_indices(y, test_size=0.20, random_state=42):
    return train_test_split(
        np.arange(len(y)),
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

def evaluate_threshold(y_true, probabilities, threshold):
    predictions = (np.asarray(probabilities) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    return {
        "threshold": float(threshold),
        "accuracy": accuracy_score(y_true, predictions),
        "balanced_accuracy": balanced_accuracy_score(y_true, predictions),
        "precision": precision_score(y_true, predictions, zero_division=0),
        "recall": recall_score(y_true, predictions, zero_division=0),
        "f1": f1_score(y_true, predictions, zero_division=0),
        "f2": fbeta_score(y_true, predictions, beta=2, zero_division=0),
        "specificity": specificity,
        "alert_rate": float(predictions.mean()),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
    }

def threshold_sweep(y_true, probabilities, start=0.05, stop=0.95, step=0.01):
    thresholds = np.arange(start, stop + step / 2, step)
    return pd.DataFrame([
        evaluate_threshold(y_true, probabilities, threshold)
        for threshold in thresholds
    ])

def evaluate_holdout(y_true, probabilities, threshold,
                     model_name="XGBoost", policy="Balanced"):
    metrics = evaluate_threshold(y_true, probabilities, threshold)
    metrics.update({
        "model": model_name,
        "policy": policy,
        "roc_auc": roc_auc_score(y_true, probabilities),
        "pr_auc": average_precision_score(y_true, probabilities),
    })
    return metrics
