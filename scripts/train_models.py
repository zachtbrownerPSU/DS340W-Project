import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


CONTEXT_CATEGORICAL = [
    "FieldType",
    "StadiumType_clean",
    "Weather_clean",
    "RosterPosition",
    "Position",
    "PositionGroup",
]

CONTEXT_NUMERIC = [
    "Temperature",
    "PlayerDay",
    "PlayerGame",
    "games_before",
    "days_since_previous_game",
    "prior_injury_count",
]

HISTORY_NUMERIC = [
    "prior_plays",
    "prior_tracked_plays",
    "prior_distance",
    "prior_mean_speed",
    "prior_max_speed",
    "prior_high_speed_distance",
    "prior_direction_changes",
    "prior_acceleration_max",
    "prior_deceleration_min",
    "prior_3_plays",
    "prior_3_tracked_plays",
    "prior_3_distance",
    "prior_3_mean_speed",
    "prior_3_max_speed",
    "prior_3_high_speed_distance",
    "prior_3_direction_changes",
    "prior_3_acceleration_max",
    "prior_3_deceleration_min",
]


def build_preprocessor(categorical, numeric):
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    numeric_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    return ColumnTransformer(
        [
            ("categorical", categorical_pipeline, categorical),
            ("numeric", numeric_pipeline, numeric),
        ]
    )


def logistic_pipeline(categorical, numeric):
    return Pipeline(
        [
            ("preprocess", build_preprocessor(categorical, numeric)),
            (
                "model",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=3000,
                    solver="liblinear",
                    random_state=340,
                ),
            ),
        ]
    )


def forest_pipeline(categorical, numeric):
    return Pipeline(
        [
            ("preprocess", build_preprocessor(categorical, numeric)),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=600,
                    min_samples_leaf=3,
                    max_features="sqrt",
                    class_weight="balanced_subsample",
                    n_jobs=-1,
                    random_state=340,
                ),
            ),
        ]
    )


def xgboost_pipeline(categorical, numeric, scale_pos_weight, device):
    from xgboost import XGBClassifier

    return Pipeline(
        [
            ("preprocess", build_preprocessor(categorical, numeric)),
            (
                "model",
                XGBClassifier(
                    n_estimators=700,
                    max_depth=4,
                    learning_rate=0.03,
                    subsample=0.85,
                    colsample_bytree=0.85,
                    min_child_weight=3,
                    reg_lambda=1.0,
                    scale_pos_weight=scale_pos_weight,
                    eval_metric="logloss",
                    tree_method="hist",
                    device=device,
                    random_state=340,
                ),
            ),
        ]
    )


def choose_threshold(labels, probabilities):
    candidates = np.unique(probabilities)
    best_threshold = 0.5
    best_score = -1.0
    for threshold in candidates:
        predictions = probabilities >= threshold
        score = f1_score(labels, predictions, zero_division=0)
        if score > best_score:
            best_score = score
            best_threshold = float(threshold)
    return best_threshold


def calculate_metrics(labels, probabilities, threshold):
    predictions = (probabilities >= threshold).astype(int)
    matrix = confusion_matrix(labels, predictions, labels=[0, 1])
    return {
        "rows": int(len(labels)),
        "positive_rows": int(np.sum(labels)),
        "threshold": float(threshold),
        "average_precision": float(average_precision_score(labels, probabilities)),
        "roc_auc": float(roc_auc_score(labels, probabilities)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "brier_score": float(brier_score_loss(labels, probabilities)),
        "confusion_matrix": matrix.tolist(),
    }


def save_importance(pipeline, path):
    preprocessor = pipeline.named_steps["preprocess"]
    model = pipeline.named_steps["model"]
    feature_names = preprocessor.get_feature_names_out()
    if hasattr(model, "coef_"):
        values = model.coef_[0]
    elif hasattr(model, "feature_importances_"):
        values = model.feature_importances_
    else:
        return
    importance = pd.DataFrame({"feature": feature_names, "importance": values})
    importance["absolute_importance"] = importance["importance"].abs()
    importance = importance.sort_values("absolute_importance", ascending=False)
    importance.to_csv(path, index=False)


def model_specs(include_xgboost, device, labels):
    all_numeric = CONTEXT_NUMERIC + HISTORY_NUMERIC
    no_surface = [column for column in CONTEXT_CATEGORICAL if column != "FieldType"]
    negatives = int((labels == 0).sum())
    positives = int((labels == 1).sum())
    ratio = negatives / positives
    specs = {
        "surface_logistic": (
            logistic_pipeline(["FieldType"], []),
            ["FieldType"],
        ),
        "context_logistic": (
            logistic_pipeline(CONTEXT_CATEGORICAL, CONTEXT_NUMERIC),
            CONTEXT_CATEGORICAL + CONTEXT_NUMERIC,
        ),
        "full_logistic": (
            logistic_pipeline(CONTEXT_CATEGORICAL, all_numeric),
            CONTEXT_CATEGORICAL + all_numeric,
        ),
        "full_logistic_no_surface": (
            logistic_pipeline(no_surface, all_numeric),
            no_surface + all_numeric,
        ),
        "full_random_forest": (
            forest_pipeline(CONTEXT_CATEGORICAL, all_numeric),
            CONTEXT_CATEGORICAL + all_numeric,
        ),
        "full_random_forest_no_surface": (
            forest_pipeline(no_surface, all_numeric),
            no_surface + all_numeric,
        ),
    }
    if include_xgboost:
        specs["full_xgboost"] = (
            xgboost_pipeline(CONTEXT_CATEGORICAL, all_numeric, ratio, device),
            CONTEXT_CATEGORICAL + all_numeric,
        )
        specs["full_xgboost_no_surface"] = (
            xgboost_pipeline(no_surface, all_numeric, ratio, device),
            no_surface + all_numeric,
        )
    return specs


def run_model(name, pipeline, features, training, evaluation, output_dir):
    train_x = training[features]
    train_y = training["injury"].to_numpy()
    train_groups = training["PlayerKey"].to_numpy()
    eval_x = evaluation[features]
    eval_y = evaluation["injury"].to_numpy()

    folds = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=340)
    out_of_fold = cross_val_predict(
        clone(pipeline),
        train_x,
        train_y,
        groups=train_groups,
        cv=folds,
        method="predict_proba",
        n_jobs=1,
    )[:, 1]
    threshold = choose_threshold(train_y, out_of_fold)

    pipeline.fit(train_x, train_y)
    probabilities = pipeline.predict_proba(eval_x)[:, 1]
    metrics = calculate_metrics(eval_y, probabilities, threshold)

    predictions = evaluation[["PlayerKey", "GameID", "split", "injury"]].copy()
    predictions["probability"] = probabilities
    predictions["prediction"] = (probabilities >= threshold).astype(int)
    predictions.to_csv(output_dir / f"{name}_predictions.csv", index=False)
    joblib.dump(pipeline, output_dir / f"{name}.joblib")
    save_importance(pipeline, output_dir / f"{name}_feature_importance.csv")
    print(
        f"model={name} average_precision={metrics['average_precision']:.4f} "
        f"recall={metrics['recall']:.4f} precision={metrics['precision']:.4f}"
    )
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--table", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--include-xgboost", action="store_true")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--final-model")
    args = parser.parse_args()

    data = pd.read_csv(args.table)
    missing = [
        column
        for column in CONTEXT_CATEGORICAL + CONTEXT_NUMERIC + HISTORY_NUMERIC
        if column not in data.columns
    ]
    if missing:
        raise ValueError(f"Modeling table is missing required columns: {missing}")

    if args.final_model:
        training = data[data["split"].isin(["train", "test"])].copy()
        evaluation = data[data["split"].eq("validation")].copy()
        mode = "final_validation"
    else:
        training = data[data["split"].eq("train")].copy()
        evaluation = data[data["split"].eq("test")].copy()
        mode = "model_development"

    specs = model_specs(args.include_xgboost, args.device, training["injury"].to_numpy())
    if args.final_model:
        if args.final_model not in specs:
            raise ValueError(f"Unknown final model: {args.final_model}")
        specs = {args.final_model: specs[args.final_model]}

    args.output_dir.mkdir(parents=True, exist_ok=True)
    prevalence = float(training["injury"].mean())
    metrics = {
        "run_mode": mode,
        "training_rows": int(len(training)),
        "evaluation_rows": int(len(evaluation)),
        "training_players": int(training["PlayerKey"].nunique()),
        "evaluation_players": int(evaluation["PlayerKey"].nunique()),
        "training_prevalence": prevalence,
        "models": {
            "prevalence_baseline": calculate_metrics(
                evaluation["injury"].to_numpy(),
                np.full(len(evaluation), prevalence),
                0.5,
            )
        },
    }

    for name, (pipeline, features) in specs.items():
        metrics["models"][name] = run_model(
            name,
            pipeline,
            features,
            training,
            evaluation,
            args.output_dir,
        )

    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    rows = []
    for name, values in metrics["models"].items():
        row = {"model": name}
        row.update({key: value for key, value in values.items() if key != "confusion_matrix"})
        rows.append(row)
    pd.DataFrame(rows).to_csv(args.output_dir / "metrics.csv", index=False)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
