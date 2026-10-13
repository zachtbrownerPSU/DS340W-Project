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
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


CATEGORICAL = [
    "FieldType",
    "StadiumType_clean",
    "Weather_clean",
    "RosterPosition",
    "Position",
    "PositionGroup",
    "PlayType",
]

CONTEXT_NUMERIC = [
    "Temperature",
    "PlayerDay",
    "PlayerGame",
    "PlayerGamePlay",
    "plays_completed",
]

MOVEMENT_NUMERIC = [
    "distance_total",
    "speed_mean",
    "speed_max",
    "speed_std",
    "acceleration_max",
    "deceleration_min",
    "direction_change_total",
    "sharp_direction_changes",
    "high_speed_distance",
    "duration_seconds",
    "recent_5_distance_total",
    "recent_5_speed_mean",
    "recent_5_speed_max",
    "recent_5_acceleration_max",
    "recent_5_deceleration_min",
    "recent_5_sharp_direction_changes",
    "recent_5_high_speed_distance",
    "game_to_date_distance_total",
    "game_to_date_sharp_direction_changes",
    "game_to_date_high_speed_distance",
]


def preprocessor(categorical, numeric):
    categories = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    numbers = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    return ColumnTransformer(
        [("categorical", categories, categorical), ("numeric", numbers, numeric)]
    )


def logistic(categorical, numeric):
    return Pipeline(
        [
            ("preprocess", preprocessor(categorical, numeric)),
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


def forest(categorical, numeric):
    return Pipeline(
        [
            ("preprocess", preprocessor(categorical, numeric)),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=400,
                    max_depth=16,
                    min_samples_leaf=5,
                    max_features="sqrt",
                    class_weight="balanced_subsample",
                    n_jobs=-1,
                    random_state=340,
                ),
            ),
        ]
    )


def xgboost(categorical, numeric, scale_pos_weight, device):
    from xgboost import XGBClassifier

    return Pipeline(
        [
            ("preprocess", preprocessor(categorical, numeric)),
            (
                "model",
                XGBClassifier(
                    n_estimators=900,
                    max_depth=5,
                    learning_rate=0.025,
                    subsample=0.85,
                    colsample_bytree=0.85,
                    min_child_weight=3,
                    reg_lambda=2.0,
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
    order = np.argsort(probabilities)[::-1]
    sorted_labels = labels[order]
    true_positives = np.cumsum(sorted_labels)
    predicted_positives = np.arange(1, len(labels) + 1)
    total_positives = sorted_labels.sum()
    precision = true_positives / predicted_positives
    recall = true_positives / total_positives
    scores = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    best = int(np.nanargmax(scores))
    return float(probabilities[order[best]])


def metrics(labels, probabilities, threshold, evaluation):
    predictions = (probabilities >= threshold).astype(int)
    matrix = confusion_matrix(labels, predictions, labels=[0, 1])
    positive_games = evaluation.loc[labels == 1, ["PlayerKey", "GameID"]].drop_duplicates()
    alerted_games = evaluation.loc[
        (labels == 1) & (predictions == 1), ["PlayerKey", "GameID"]
    ].drop_duplicates()
    negative_rows = int((labels == 0).sum())
    false_alerts = int(((labels == 0) & (predictions == 1)).sum())
    return {
        "rows": int(len(labels)),
        "positive_rows": int(labels.sum()),
        "positive_injury_games": int(len(positive_games)),
        "threshold": float(threshold),
        "average_precision": float(average_precision_score(labels, probabilities)),
        "roc_auc": float(roc_auc_score(labels, probabilities)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "brier_score": float(brier_score_loss(labels, probabilities)),
        "injury_game_alert_rate": float(len(alerted_games) / len(positive_games)),
        "false_alerts_per_1000_negative_plays": float(false_alerts * 1000 / negative_rows),
        "confusion_matrix": matrix.tolist(),
    }


def specifications(labels, device):
    all_numeric = CONTEXT_NUMERIC + MOVEMENT_NUMERIC
    no_surface = [column for column in CATEGORICAL if column != "FieldType"]
    ratio = float((labels == 0).sum() / (labels == 1).sum())
    return {
        "surface_logistic": (logistic(["FieldType"], []), ["FieldType"]),
        "full_logistic": (
            logistic(CATEGORICAL, all_numeric),
            CATEGORICAL + all_numeric,
        ),
        "full_logistic_no_surface": (
            logistic(no_surface, all_numeric),
            no_surface + all_numeric,
        ),
        "full_random_forest": (
            forest(CATEGORICAL, all_numeric),
            CATEGORICAL + all_numeric,
        ),
        "full_random_forest_no_surface": (
            forest(no_surface, all_numeric),
            no_surface + all_numeric,
        ),
        "full_xgboost": (
            xgboost(CATEGORICAL, all_numeric, ratio, device),
            CATEGORICAL + all_numeric,
        ),
        "full_xgboost_no_surface": (
            xgboost(no_surface, all_numeric, ratio, device),
            no_surface + all_numeric,
        ),
    }


def out_of_fold_probabilities(pipeline, features, data, target, folds):
    labels = data[target].to_numpy()
    groups = data["PlayerKey"].to_numpy()
    probabilities = np.zeros(len(data), dtype=float)
    splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=340)
    for train_index, test_index in splitter.split(data[features], labels, groups):
        fitted = clone(pipeline)
        fitted.fit(data.iloc[train_index][features], labels[train_index])
        probabilities[test_index] = fitted.predict_proba(
            data.iloc[test_index][features]
        )[:, 1]
    return probabilities


def run_model(name, pipeline, features, training, evaluation, target, folds, output_dir):
    train_labels = training[target].to_numpy()
    evaluation_labels = evaluation[target].to_numpy()
    out_of_fold = out_of_fold_probabilities(
        pipeline, features, training, target, folds
    )
    threshold = choose_threshold(train_labels, out_of_fold)
    pipeline.fit(training[features], train_labels)
    probabilities = pipeline.predict_proba(evaluation[features])[:, 1]
    values = metrics(evaluation_labels, probabilities, threshold, evaluation)
    predictions = evaluation[
        [
            "PlayerKey",
            "GameID",
            "PlayKey",
            "split",
            "injury_play_number",
            "plays_until_injury",
            target,
        ]
    ].copy()
    predictions["probability"] = probabilities
    predictions["prediction"] = (probabilities >= threshold).astype("int8")
    predictions.to_csv(output_dir / f"{name}_predictions.csv", index=False)
    joblib.dump(pipeline, output_dir / f"{name}.joblib", compress=3)
    print(
        f"target={target} model={name} average_precision={values['average_precision']:.5f} "
        f"recall={values['recall']:.5f} precision={values['precision']:.5f}"
    )
    return values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--table", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--windows", type=int, nargs="+", default=[5, 10, 20])
    args = parser.parse_args()

    data = pd.read_csv(args.table)
    required = CATEGORICAL + CONTEXT_NUMERIC + MOVEMENT_NUMERIC
    missing = [column for column in required if column not in data.columns]
    if missing:
        raise ValueError(f"Play sequence table is missing required columns: {missing}")

    training = data[data["split"].eq("train")].reset_index(drop=True)
    evaluation = data[data["split"].eq("test")].reset_index(drop=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "training_rows": int(len(training)),
        "evaluation_rows": int(len(evaluation)),
        "training_players": int(training["PlayerKey"].nunique()),
        "evaluation_players": int(evaluation["PlayerKey"].nunique()),
        "folds": args.folds,
        "device": args.device,
        "targets": {},
    }

    for window in args.windows:
        target = f"injury_within_{window}_plays"
        target_dir = args.output_dir / f"within_{window}_plays"
        target_dir.mkdir(parents=True, exist_ok=True)
        labels = training[target].to_numpy()
        if labels.sum() == 0:
            raise RuntimeError(f"Training data have no positive rows for {target}")
        target_metrics = {}
        prevalence = float(labels.mean())
        evaluation_labels = evaluation[target].to_numpy()
        target_metrics["prevalence_baseline"] = metrics(
            evaluation_labels,
            np.full(len(evaluation), prevalence),
            0.5,
            evaluation,
        )
        for name, (pipeline, features) in specifications(labels, args.device).items():
            target_metrics[name] = run_model(
                name,
                pipeline,
                features,
                training,
                evaluation,
                target,
                args.folds,
                target_dir,
            )
        rows = []
        for name, values in target_metrics.items():
            row = {"model": name}
            row.update(
                {key: value for key, value in values.items() if key != "confusion_matrix"}
            )
            rows.append(row)
        pd.DataFrame(rows).sort_values(
            "average_precision", ascending=False
        ).to_csv(target_dir / "metrics.csv", index=False)
        summary["targets"][target] = target_metrics

    (args.output_dir / "metrics.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(f"saved={args.output_dir} targets={len(args.windows)}")


if __name__ == "__main__":
    main()
