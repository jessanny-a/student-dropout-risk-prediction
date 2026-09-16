import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)

from .config import MIN_GROUP_SIZE, MIN_NEGATIVE_CASES, MIN_POSITIVE_CASES

AUDIT_DIMENSIONS = [
    "gender_group",
    "age_group",
    "displaced_group",
    "scholarship_group",
    "debtor_group",
    "tuition_status_group",
    "financial_risk_group",
]

def safe_ratio(numerator, denominator):
    return np.nan if denominator == 0 else numerator / denominator

def add_group_labels(df):
    out = df.copy()
    out["age_group"] = pd.cut(
        out["age_at_enrollment"],
        bins=[16, 20, 24, 29, 39, np.inf],
        labels=["17-20", "21-24", "25-29", "30-39", "40+"],
        include_lowest=True,
    )
    out["gender_group"] = out["gender"].astype(str).map(
        lambda x: f"Gender code {x}"
    )
    out["displaced_group"] = out["displaced"].astype(str).map(
        lambda x: f"Displaced code {x}"
    )
    out["scholarship_group"] = out["scholarship_holder"].astype(str).map(
        lambda x: f"Scholarship code {x}"
    )
    out["debtor_group"] = out["debtor"].astype(str).map(
        lambda x: f"Debtor code {x}"
    )
    out["tuition_status_group"] = out["tuition_fees_up_to_date"].astype(str).map(
        lambda x: f"Tuition-status code {x}"
    )
    out["financial_risk_group"] = out["financial_risk_count"].astype(str).map(
        lambda x: f"Financial-risk count {x}"
    )
    return out

def build_fairness_dataset(X_holdout, y_true, predictions,
                           probabilities, threshold):
    out = X_holdout.copy()
    out["actual_dropout"] = np.asarray(y_true)
    out["predicted_dropout"] = np.asarray(predictions)
    out["dropout_probability"] = np.asarray(probabilities)
    out["decision_threshold"] = float(threshold)
    return add_group_labels(out)

def subgroup_metrics(data, group_column):
    rows = []
    for group_value, group_data in data.groupby(
        group_column, observed=False, dropna=False
    ):
        y_true = group_data["actual_dropout"].astype(int).to_numpy()
        y_pred = group_data["predicted_dropout"].astype(int).to_numpy()
        y_prob = group_data["dropout_probability"].astype(float).to_numpy()

        tn, fp, fn, tp = confusion_matrix(
            y_true, y_pred, labels=[0, 1]
        ).ravel()

        n = len(group_data)
        actual_dropouts = tp + fn
        actual_graduates = tn + fp
        dropout_prevalence = y_true.mean() if n else np.nan
        alert_rate = y_pred.mean() if n else np.nan

        precision = safe_ratio(tp, tp + fp)
        recall = safe_ratio(tp, tp + fn)
        fnr = safe_ratio(fn, fn + tp)
        specificity = safe_ratio(tn, tn + fp)
        fpr = safe_ratio(fp, fp + tn)

        accuracy = accuracy_score(y_true, y_pred) if n else np.nan
        balanced_accuracy = (
            balanced_accuracy_score(y_true, y_pred)
            if len(np.unique(y_true)) == 2 else np.nan
        )
        f1 = (
            f1_score(y_true, y_pred, zero_division=0)
            if actual_dropouts > 0 else np.nan
        )
        roc_auc = (
            roc_auc_score(y_true, y_prob)
            if len(np.unique(y_true)) == 2 else np.nan
        )
        pr_auc = (
            average_precision_score(y_true, y_prob)
            if actual_dropouts > 0 else np.nan
        )

        mean_predicted_risk = y_prob.mean()
        calibration_gap = mean_predicted_risk - dropout_prevalence

        rows.append({
            "audit_dimension": group_column,
            "group": group_value,
            "n": n,
            "actual_dropouts": actual_dropouts,
            "actual_graduates": actual_graduates,
            "dropout_prevalence": dropout_prevalence,
            "alert_rate": alert_rate,
            "precision": precision,
            "recall": recall,
            "false_negative_rate": fnr,
            "specificity": specificity,
            "false_positive_rate": fpr,
            "accuracy": accuracy,
            "balanced_accuracy": balanced_accuracy,
            "f1": f1,
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
            "mean_predicted_risk": mean_predicted_risk,
            "calibration_gap": calibration_gap,
            "true_positives": tp,
            "false_negatives": fn,
            "true_negatives": tn,
            "false_positives": fp,
        })
    return pd.DataFrame(rows)

def mark_sufficient_support(
    table,
    min_group_size=MIN_GROUP_SIZE,
    min_positive_cases=MIN_POSITIVE_CASES,
    min_negative_cases=MIN_NEGATIVE_CASES,
):
    out = table.copy()
    out["sufficient_support"] = (
        (out["n"] >= min_group_size)
        & (out["actual_dropouts"] >= min_positive_cases)
        & (out["actual_graduates"] >= min_negative_cases)
    )
    return out

def run_primary_audits(fairness_df):
    return {
        dimension: mark_sufficient_support(
            subgroup_metrics(fairness_df, dimension)
        )
        for dimension in AUDIT_DIMENSIONS
    }

def disparity_gap_summary(fairness_tables):
    rows = []
    for dimension, table in fairness_tables.items():
        eligible = table[table["sufficient_support"]].copy()
        if len(eligible) < 2:
            continue
        rows.append({
            "audit_dimension": dimension,
            "eligible_groups": len(eligible),
            "recall_gap": eligible["recall"].max() - eligible["recall"].min(),
            "false_negative_rate_gap": (
                eligible["false_negative_rate"].max()
                - eligible["false_negative_rate"].min()
            ),
            "false_positive_rate_gap": (
                eligible["false_positive_rate"].max()
                - eligible["false_positive_rate"].min()
            ),
            "precision_gap": (
                eligible["precision"].max() - eligible["precision"].min()
            ),
            "alert_rate_gap": (
                eligible["alert_rate"].max() - eligible["alert_rate"].min()
            ),
            "calibration_gap_range": (
                eligible["calibration_gap"].max()
                - eligible["calibration_gap"].min()
            ),
        })
    return (
        pd.DataFrame(rows)
        .sort_values("recall_gap", ascending=False)
        .reset_index(drop=True)
    )

def formal_fairness_metrics(fairness_tables):
    rows = []
    for dimension, table in fairness_tables.items():
        eligible = table[table["sufficient_support"]].copy()

        if len(eligible) < 2:
            rows.append({
                "audit_dimension": dimension,
                "eligible_groups": len(eligible),
                "demographic_parity_difference": np.nan,
                "disparate_impact_ratio": np.nan,
                "equal_opportunity_difference": np.nan,
                "equalised_odds_difference": np.nan,
                "tpr_gap": np.nan,
                "fpr_gap": np.nan,
                "assessment_status": "Insufficient comparable support",
            })
            continue

        alert = eligible["alert_rate"].astype(float)
        tpr = eligible["recall"].astype(float)
        fpr = eligible["false_positive_rate"].astype(float)

        dp_diff = alert.max() - alert.min()
        di_ratio = alert.min() / alert.max() if alert.max() > 0 else np.nan
        tpr_gap = tpr.max() - tpr.min()
        fpr_gap = fpr.max() - fpr.min()

        rows.append({
            "audit_dimension": dimension,
            "eligible_groups": len(eligible),
            "demographic_parity_difference": dp_diff,
            "disparate_impact_ratio": di_ratio,
            "equal_opportunity_difference": tpr_gap,
            "equalised_odds_difference": max(tpr_gap, fpr_gap),
            "tpr_gap": tpr_gap,
            "fpr_gap": fpr_gap,
            "assessment_status": "Comparable",
        })

    return (
        pd.DataFrame(rows)
        .sort_values(
            "equalised_odds_difference",
            ascending=False,
            na_position="last",
        )
        .reset_index(drop=True)
    )
