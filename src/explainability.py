import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.special import expit
from sklearn.inspection import PartialDependenceDisplay
from sklearn.pipeline import Pipeline

def get_preprocessor_and_estimator(final_pipeline):
    return Pipeline(final_pipeline.steps[:-1]), final_pipeline.named_steps["model"]

def transform_with_feature_names(final_pipeline, X):
    preprocessor, _ = get_preprocessor_and_estimator(final_pipeline)
    transformed = preprocessor.transform(X)
    names = np.asarray(preprocessor.get_feature_names_out())
    if transformed.shape[1] != len(names):
        raise ValueError("Transformed feature count does not match extracted names.")
    return transformed, names

def xgb_shap_contributions(final_pipeline, X):
    preprocessor, estimator = get_preprocessor_and_estimator(final_pipeline)
    transformed = preprocessor.transform(X)
    names = np.asarray(preprocessor.get_feature_names_out())
    matrix = xgb.DMatrix(transformed)
    with_bias = estimator.get_booster().predict(matrix, pred_contribs=True)
    return {
        "shap_values": with_bias[:, :-1],
        "base_values": with_bias[:, -1],
        "feature_names": names,
    }

def shap_reconstruction_error(base_values, shap_values, reference_probabilities):
    reconstructed_margin = base_values + shap_values.sum(axis=1)
    reconstructed_probability = expit(reconstructed_margin)
    return float(np.max(
        np.abs(reconstructed_probability - np.asarray(reference_probabilities))
    ))

def map_to_raw_feature(transformed_name, raw_feature_names):
    clean_name = (
        transformed_name.split("__", 1)[1]
        if "__" in transformed_name else transformed_name
    )
    if clean_name in raw_feature_names:
        return clean_name
    for raw_feature in sorted(raw_feature_names, key=len, reverse=True):
        if clean_name.startswith(raw_feature + "_"):
            return raw_feature
    return clean_name

def aggregate_shap_to_raw(shap_values, transformed_feature_names, raw_feature_names):
    transformed_to_raw = pd.Series(
        [map_to_raw_feature(name, list(raw_feature_names))
         for name in transformed_feature_names],
        index=transformed_feature_names,
    )
    shap_df = pd.DataFrame(shap_values, columns=transformed_feature_names)
    grouped = pd.DataFrame(index=shap_df.index)
    for raw_feature in sorted(transformed_to_raw.unique()):
        matching = transformed_to_raw[
            transformed_to_raw == raw_feature
        ].index.tolist()
        grouped[raw_feature] = shap_df[matching].sum(axis=1)
    global_importance = pd.DataFrame({
        "raw_feature": grouped.columns,
        "mean_abs_shap": grouped.abs().mean(axis=0).values,
    }).sort_values("mean_abs_shap", ascending=False, ignore_index=True)
    return grouped, global_importance

def local_shap_explanation(row_position, X_raw, shap_values,
                           transformed_feature_names, top_n=12):
    raw_feature_names = list(X_raw.columns)
    contribution_table = pd.DataFrame({
        "transformed_feature": transformed_feature_names,
        "shap_contribution": shap_values[row_position],
    })
    contribution_table["raw_feature"] = contribution_table[
        "transformed_feature"
    ].apply(lambda name: map_to_raw_feature(name, raw_feature_names))
    local_raw = contribution_table.groupby(
        "raw_feature", as_index=False
    )["shap_contribution"].sum()
    local_raw["absolute_contribution"] = local_raw["shap_contribution"].abs()
    local_raw["direction"] = np.where(
        local_raw["shap_contribution"] > 0,
        "Increases predicted dropout risk",
        "Decreases predicted dropout risk",
    )
    original_row = X_raw.iloc[row_position]
    local_raw["raw_value"] = local_raw["raw_feature"].map(original_row.to_dict())
    return (
        local_raw.sort_values("absolute_contribution", ascending=False)
        .head(top_n).reset_index(drop=True)
    )

def plot_pdp(final_pipeline, X_train, feature="sem1_completion_rate", ax=None):
    return PartialDependenceDisplay.from_estimator(
        final_pipeline, X_train, features=[feature], kind="average",
        response_method="predict_proba", grid_resolution=50,
        percentiles=(0.05, 0.95), ax=ax,
    )

def plot_ice(final_pipeline, X_train, feature="sem1_completion_rate",
             subsample=100, random_state=42, ax=None):
    return PartialDependenceDisplay.from_estimator(
        final_pipeline, X_train, features=[feature], kind="individual",
        response_method="predict_proba", subsample=subsample,
        random_state=random_state, grid_resolution=50,
        percentiles=(0.05, 0.95), ax=ax,
    )
