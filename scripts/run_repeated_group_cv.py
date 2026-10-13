import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

from train_models import model_specs


MODEL_NAMES = [
    "surface_logistic",
    "context_logistic",
    "full_logistic",
    "full_logistic_no_surface",
    "full_random_forest",
    "full_random_forest_no_surface",
    "full_xgboost",
    "full_xgboost_no_surface",
]


def score(labels, probabilities):
    return {
        "average_precision": float(average_precision_score(labels, probabilities)),
        "roc_auc": float(roc_auc_score(labels, probabilities)),
        "brier_score": float(brier_score_loss(labels, probabilities)),
    }


def summarize(metrics):
    rows = []
    for model, group in metrics.groupby("model", sort=False):
        for metric in ["average_precision", "roc_auc", "brier_score"]:
            values = group[metric]
            rows.append(
                {
                    "model": model,
                    "metric": metric,
                    "mean": values.mean(),
                    "standard_deviation": values.std(ddof=1),
                    "lower_2_5": values.quantile(0.025),
                    "median": values.median(),
                    "upper_97_5": values.quantile(0.975),
                    "fold_results": len(values),
                }
            )
    return pd.DataFrame(rows)


def paired_surface_differences(metrics):
    pairs = [
        ("full_logistic", "full_logistic_no_surface"),
        ("full_random_forest", "full_random_forest_no_surface"),
        ("full_xgboost", "full_xgboost_no_surface"),
    ]
    rows = []
    keys = ["repeat", "fold"]
    for with_surface, without_surface in pairs:
        left = metrics[metrics["model"].eq(with_surface)]
        right = metrics[metrics["model"].eq(without_surface)]
        merged = left.merge(right, on=keys, suffixes=("_with", "_without"))
        for metric in ["average_precision", "roc_auc", "brier_score"]:
            values = merged[f"{metric}_with"] - merged[f"{metric}_without"]
            rows.append(
                {
                    "model_with_surface": with_surface,
                    "model_without_surface": without_surface,
                    "metric": metric,
                    "mean_difference": values.mean(),
                    "standard_deviation": values.std(ddof=1),
                    "lower_2_5": values.quantile(0.025),
                    "median": values.median(),
                    "upper_97_5": values.quantile(0.975),
                    "fraction_above_zero": float((values > 0).mean()),
                    "paired_fold_results": len(values),
                }
            )
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--table", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--seed", type=int, default=340)
    args = parser.parse_args()

    data = pd.read_csv(args.table)
    data = data[data["split"].isin(["train", "test"])].reset_index(drop=True)
    labels = data["injury"].to_numpy()
    groups = data["PlayerKey"].to_numpy()
    specs = model_specs(True, args.device, labels)
    specs = {name: specs[name] for name in MODEL_NAMES}
    metric_rows = []
    prediction_rows = []

    for repetition in range(args.repeats):
        splitter = StratifiedGroupKFold(
            n_splits=args.folds,
            shuffle=True,
            random_state=args.seed + repetition,
        )
        fold_indices = list(splitter.split(data, labels, groups))
        for name, (pipeline, features) in specs.items():
            for fold, (train_index, test_index) in enumerate(fold_indices, start=1):
                fitted = clone(pipeline)
                fitted.fit(data.iloc[train_index][features], labels[train_index])
                probabilities = fitted.predict_proba(data.iloc[test_index][features])[:, 1]
                values = score(labels[test_index], probabilities)
                metric_rows.append(
                    {
                        "repeat": repetition + 1,
                        "fold": fold,
                        "model": name,
                        "training_rows": len(train_index),
                        "evaluation_rows": len(test_index),
                        "evaluation_players": data.iloc[test_index]["PlayerKey"].nunique(),
                        "positive_rows": int(labels[test_index].sum()),
                        **values,
                    }
                )
                fold_predictions = data.iloc[test_index][
                    ["PlayerKey", "GameID", "injury"]
                ].copy()
                fold_predictions.insert(0, "model", name)
                fold_predictions.insert(0, "fold", fold)
                fold_predictions.insert(0, "repeat", repetition + 1)
                fold_predictions["probability"] = probabilities
                prediction_rows.append(fold_predictions)
                print(
                    f"repeat={repetition + 1} fold={fold} model={name} "
                    f"average_precision={values['average_precision']:.5f}"
                )

    metrics = pd.DataFrame(metric_rows)
    predictions = pd.concat(prediction_rows, ignore_index=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(args.output_dir / "fold_metrics.csv", index=False)
    predictions.to_csv(args.output_dir / "out_of_fold_predictions.csv", index=False)
    summarize(metrics).to_csv(args.output_dir / "metric_summary.csv", index=False)
    paired_surface_differences(metrics).to_csv(
        args.output_dir / "surface_paired_differences.csv", index=False
    )
    print(
        f"saved={args.output_dir} repeats={args.repeats} folds={args.folds} "
        f"players={data['PlayerKey'].nunique()} validation_used=false"
    )


if __name__ == "__main__":
    main()
