import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from xgboost import XGBClassifier

from train_models import (
    CONTEXT_CATEGORICAL,
    CONTEXT_NUMERIC,
    HISTORY_NUMERIC,
    build_preprocessor,
)


def candidate_parameters(generator):
    return {
        "n_estimators": int(generator.choice([300, 500, 700, 1000, 1400])),
        "max_depth": int(generator.integers(2, 8)),
        "learning_rate": float(10 ** generator.uniform(-2.0, -0.82)),
        "subsample": float(generator.uniform(0.6, 1.0)),
        "colsample_bytree": float(generator.uniform(0.6, 1.0)),
        "min_child_weight": int(generator.choice([1, 2, 3, 5, 8, 12])),
        "reg_lambda": float(10 ** generator.uniform(-1.0, 1.3)),
        "reg_alpha": float(10 ** generator.uniform(-3.0, 0.7)),
        "gamma": float(generator.choice([0.0, 0.1, 0.25, 0.5, 1.0, 2.0])),
    }


def pipeline(parameters, categorical, numeric, scale_pos_weight, device):
    from sklearn.pipeline import Pipeline

    model = XGBClassifier(
        **parameters,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        tree_method="hist",
        device=device,
        random_state=340,
    )
    return Pipeline(
        [("preprocess", build_preprocessor(categorical, numeric)), ("model", model)]
    )


def evaluate_candidate(model, data, features, labels, groups, folds):
    average_precision = []
    roc_auc = []
    brier = []
    for train_index, test_index in folds:
        fitted = clone(model)
        fitted.fit(data.iloc[train_index][features], labels[train_index])
        probabilities = fitted.predict_proba(data.iloc[test_index][features])[:, 1]
        average_precision.append(
            average_precision_score(labels[test_index], probabilities)
        )
        roc_auc.append(roc_auc_score(labels[test_index], probabilities))
        brier.append(brier_score_loss(labels[test_index], probabilities))
    return {
        "mean_average_precision": float(np.mean(average_precision)),
        "standard_deviation_average_precision": float(np.std(average_precision, ddof=1)),
        "mean_roc_auc": float(np.mean(roc_auc)),
        "mean_brier_score": float(np.mean(brier)),
    }


def out_of_fold(model, data, features, labels, folds):
    probabilities = np.zeros(len(data), dtype=float)
    fold_numbers = np.zeros(len(data), dtype=int)
    for fold, (train_index, test_index) in enumerate(folds, start=1):
        fitted = clone(model)
        fitted.fit(data.iloc[train_index][features], labels[train_index])
        probabilities[test_index] = fitted.predict_proba(
            data.iloc[test_index][features]
        )[:, 1]
        fold_numbers[test_index] = fold
    return probabilities, fold_numbers


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--table", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--trials", type=int, default=60)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--seed", type=int, default=340)
    args = parser.parse_args()

    data = pd.read_csv(args.table)
    data = data[data["split"].isin(["train", "test"])].reset_index(drop=True)
    labels = data["injury"].to_numpy()
    groups = data["PlayerKey"].to_numpy()
    numeric = CONTEXT_NUMERIC + HISTORY_NUMERIC
    features = CONTEXT_CATEGORICAL + numeric
    no_surface_categorical = [
        column for column in CONTEXT_CATEGORICAL if column != "FieldType"
    ]
    no_surface_features = no_surface_categorical + numeric
    scale_pos_weight = float((labels == 0).sum() / (labels == 1).sum())
    splitter = StratifiedGroupKFold(
        n_splits=args.folds, shuffle=True, random_state=args.seed
    )
    fold_indices = list(splitter.split(data, labels, groups))
    generator = np.random.default_rng(args.seed)
    rows = []

    for trial in range(1, args.trials + 1):
        parameters = candidate_parameters(generator)
        model = pipeline(
            parameters,
            CONTEXT_CATEGORICAL,
            numeric,
            scale_pos_weight,
            args.device,
        )
        scores = evaluate_candidate(
            model, data, features, labels, groups, fold_indices
        )
        rows.append({"trial": trial, **parameters, **scores})
        print(
            f"trial={trial} average_precision={scores['mean_average_precision']:.5f} "
            f"roc_auc={scores['mean_roc_auc']:.5f}"
        )

    results = pd.DataFrame(rows).sort_values(
        ["mean_average_precision", "mean_roc_auc"], ascending=False
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.output_dir / "search_results.csv", index=False)
    parameter_names = [
        "n_estimators",
        "max_depth",
        "learning_rate",
        "subsample",
        "colsample_bytree",
        "min_child_weight",
        "reg_lambda",
        "reg_alpha",
        "gamma",
    ]
    best = results.iloc[0]
    best_parameters = {name: best[name].item() for name in parameter_names}
    best_parameters["n_estimators"] = int(best_parameters["n_estimators"])
    best_parameters["max_depth"] = int(best_parameters["max_depth"])
    best_parameters["min_child_weight"] = int(best_parameters["min_child_weight"])

    comparison_rows = []
    prediction_frames = []
    models = {
        "tuned_xgboost": (
            pipeline(
                best_parameters,
                CONTEXT_CATEGORICAL,
                numeric,
                scale_pos_weight,
                args.device,
            ),
            features,
        ),
        "tuned_xgboost_no_surface": (
            pipeline(
                best_parameters,
                no_surface_categorical,
                numeric,
                scale_pos_weight,
                args.device,
            ),
            no_surface_features,
        ),
    }
    for name, (model, model_features) in models.items():
        probabilities, fold_numbers = out_of_fold(
            model, data, model_features, labels, fold_indices
        )
        comparison_rows.append(
            {
                "model": name,
                "average_precision": average_precision_score(labels, probabilities),
                "roc_auc": roc_auc_score(labels, probabilities),
                "brier_score": brier_score_loss(labels, probabilities),
            }
        )
        predictions = data[["PlayerKey", "GameID", "injury"]].copy()
        predictions.insert(0, "model", name)
        predictions["fold"] = fold_numbers
        predictions["probability"] = probabilities
        prediction_frames.append(predictions)
        model.fit(data[model_features], labels)
        joblib.dump(model, args.output_dir / f"{name}.joblib", compress=3)

    pd.DataFrame(comparison_rows).to_csv(
        args.output_dir / "best_model_oof_metrics.csv", index=False
    )
    pd.concat(prediction_frames, ignore_index=True).to_csv(
        args.output_dir / "best_model_oof_predictions.csv", index=False
    )
    (args.output_dir / "best_parameters.json").write_text(
        json.dumps(best_parameters, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"saved={args.output_dir} trials={args.trials} players={data['PlayerKey'].nunique()} "
        f"validation_used=false"
    )


if __name__ == "__main__":
    main()
