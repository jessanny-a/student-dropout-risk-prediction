from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from .config import FROZEN_XGB_PARAMS, RANDOM_STATE
from .preprocessing import create_preprocessor

TUNING_SCORING = {
    "roc_auc": "roc_auc",
    "pr_auc": "average_precision",
    "recall": "recall",
    "precision": "precision",
    "f1": "f1",
    "balanced_accuracy": "balanced_accuracy",
}

def make_cv():
    return StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

def build_model_pipeline(model, categorical_columns, numeric_columns):
    preprocessor = create_preprocessor(categorical_columns, numeric_columns)
    return Pipeline([
        ("preprocessor", preprocessor),
        ("model", clone(model)),
    ])

def baseline_models():
    return {
        "Dummy": DummyClassifier(strategy="most_frequent"),
        "Logistic Regression": LogisticRegression(
            max_iter=5000, random_state=RANDOM_STATE
        ),
        "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1
        ),
        "XGBoost": XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }

def make_logistic_search(categorical_columns, numeric_columns):
    pipeline = build_model_pipeline(
        LogisticRegression(max_iter=5000, random_state=RANDOM_STATE),
        categorical_columns,
        numeric_columns,
    )
    grid = {
        "model__C": [0.001, 0.01, 0.05, 0.1, 0.5, 1, 2, 5, 10],
        "model__penalty": ["l1", "l2"],
        "model__solver": ["liblinear"],
        "model__class_weight": [None, "balanced"],
    }
    return GridSearchCV(
        pipeline, param_grid=grid, scoring=TUNING_SCORING,
        refit="roc_auc", cv=make_cv(), n_jobs=-1,
        return_train_score=True,
    )

def make_random_forest_search(categorical_columns, numeric_columns, n_iter=40):
    pipeline = build_model_pipeline(
        RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=1),
        categorical_columns,
        numeric_columns,
    )
    distributions = {
        "model__n_estimators": [200, 300, 500, 700],
        "model__max_depth": [None, 5, 8, 12, 16, 20],
        "model__min_samples_split": [2, 5, 10, 20],
        "model__min_samples_leaf": [1, 2, 4, 8],
        "model__max_features": ["sqrt", "log2", 0.5],
        "model__class_weight": [None, "balanced", "balanced_subsample"],
    }
    return RandomizedSearchCV(
        pipeline, param_distributions=distributions, n_iter=n_iter,
        scoring=TUNING_SCORING, refit="roc_auc", cv=make_cv(),
        random_state=RANDOM_STATE, n_jobs=-1, return_train_score=True,
    )

def make_xgb_search(categorical_columns, numeric_columns, n_iter=50):
    pipeline = build_model_pipeline(
        XGBClassifier(eval_metric="logloss", random_state=RANDOM_STATE, n_jobs=1),
        categorical_columns,
        numeric_columns,
    )
    distributions = {
        "model__n_estimators": [200, 300, 500, 700],
        "model__learning_rate": [0.01, 0.03, 0.05, 0.1],
        "model__max_depth": [2, 3, 4, 5, 6],
        "model__min_child_weight": [1, 3, 5, 7],
        "model__subsample": [0.7, 0.8, 0.9, 1.0],
        "model__colsample_bytree": [0.6, 0.8, 1.0],
        "model__gamma": [0, 0.1, 0.25, 0.5],
        "model__reg_alpha": [0, 0.01, 0.1, 1],
        "model__reg_lambda": [1, 2, 5, 10],
    }
    return RandomizedSearchCV(
        pipeline, param_distributions=distributions, n_iter=n_iter,
        scoring=TUNING_SCORING, refit="roc_auc", cv=make_cv(),
        random_state=RANDOM_STATE, n_jobs=-1, return_train_score=True,
    )

def build_frozen_final_model(categorical_columns, numeric_columns):
    model = XGBClassifier(**FROZEN_XGB_PARAMS)
    return build_model_pipeline(model, categorical_columns, numeric_columns)
