import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

from .config import (
    FINAL_MODEL_NAME,
    FINAL_POLICY,
    FINAL_THRESHOLD,
    MODEL_B_CATEGORICAL,
    MODEL_B_FEATURES,
    MODEL_B_NUMERIC,
    PREDICTION_HORIZON,
    RANDOM_STATE,
)
from .data_loading import (
    build_binary_outcome,
    clean_source_data,
    load_source_data,
    validate_source_data,
)
from .evaluate import evaluate_holdout, stratified_indices
from .fairness_audit import (
    build_fairness_dataset,
    disparity_gap_summary,
    formal_fairness_metrics,
    run_primary_audits,
)
from .feature_engineering import engineer_features
from .train import build_frozen_final_model

def run(data_path: Path, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    models_dir = output_dir / "models"
    reports_dir = output_dir / "reports"
    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    df_raw = load_source_data(data_path)
    df = clean_source_data(df_raw)
    qa = validate_source_data(df)

    df_binary = build_binary_outcome(df)
    df_fe = engineer_features(df_binary)

    X = df_fe[MODEL_B_FEATURES].copy()
    y = df_fe["dropout"].copy()

    train_idx, test_idx = stratified_indices(
        y, test_size=0.20, random_state=RANDOM_STATE
    )
    X_train = X.iloc[train_idx].copy()
    X_test = X.iloc[test_idx].copy()
    y_train = y.iloc[train_idx].copy()
    y_test = y.iloc[test_idx].copy()

    overlap = set(X_train.index).intersection(set(X_test.index))
    if overlap:
        raise RuntimeError("Train/holdout overlap detected.")

    model = build_frozen_final_model(
        MODEL_B_CATEGORICAL, MODEL_B_NUMERIC
    )
    model.fit(X_train, y_train)

    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= FINAL_THRESHOLD).astype(int)

    metrics = evaluate_holdout(
        y_test,
        probabilities,
        FINAL_THRESHOLD,
        model_name=FINAL_MODEL_NAME,
        policy=FINAL_POLICY,
    )

    fairness_df = build_fairness_dataset(
        X_test,
        y_test,
        predictions,
        probabilities,
        FINAL_THRESHOLD,
    )
    fairness_tables = run_primary_audits(fairness_df)
    fairness_gaps = disparity_gap_summary(fairness_tables)
    formal_fairness = formal_fairness_metrics(fairness_tables)

    joblib.dump(model, models_dir / "final_student_dropout_xgboost.joblib")
    pd.DataFrame([metrics]).to_csv(
        reports_dir / "final_holdout_metrics.csv", index=False
    )
    fairness_gaps.to_csv(
        reports_dir / "fairness_gap_summary.csv", index=False
    )
    formal_fairness.to_csv(
        reports_dir / "formal_fairness_metrics.csv", index=False
    )

    pd.DataFrame({
        "record_index": X_test.index,
        "actual_dropout": y_test.to_numpy(),
        "dropout_probability": probabilities,
        "decision_threshold": FINAL_THRESHOLD,
        "predicted_dropout": predictions,
    }).to_csv(
        reports_dir / "final_holdout_predictions.csv", index=False
    )

    config = {
        "model_name": FINAL_MODEL_NAME,
        "prediction_horizon": PREDICTION_HORIZON,
        "decision_threshold": FINAL_THRESHOLD,
        "training_records": int(len(y_train)),
        "holdout_records": int(len(y_test)),
        "raw_feature_count": int(X.shape[1]),
        "source_qa": qa,
        "holdout_metrics": {
            key: float(value) if hasattr(value, "__float__") else value
            for key, value in metrics.items()
            if key not in {"model", "policy"}
        },
    }
    (models_dir / "final_deployment_config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )

    print("Reproduction complete.")
    print(f"Training records: {len(y_train)}")
    print(f"Holdout records: {len(y_test)}")
    print(f"Frozen threshold: {FINAL_THRESHOLD}")
    print(pd.Series(metrics).to_string())

def main():
    parser = argparse.ArgumentParser(
        description="Reproduce the frozen final student-dropout model."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/raw/Predict Student Dropout.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reproduction_output"),
    )
    args = parser.parse_args()
    run(args.data, args.output_dir)

if __name__ == "__main__":
    main()
