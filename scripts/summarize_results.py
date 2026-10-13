import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


def sampled_rows(frame, sampled_players):
    groups = {key: group for key, group in frame.groupby("PlayerKey", sort=False)}
    return pd.concat([groups[key] for key in sampled_players], ignore_index=True)


def bootstrap_scores(frame, repetitions, seed):
    players = frame["PlayerKey"].drop_duplicates().to_numpy()
    generator = np.random.default_rng(seed)
    rows = []
    for repetition in range(repetitions):
        sample = sampled_rows(frame, generator.choice(players, size=len(players), replace=True))
        labels = sample["injury"].to_numpy()
        probabilities = sample["probability"].to_numpy()
        if np.unique(labels).size < 2:
            continue
        rows.append(
            {
                "repetition": repetition,
                "average_precision": average_precision_score(labels, probabilities),
                "roc_auc": roc_auc_score(labels, probabilities),
                "brier_score": brier_score_loss(labels, probabilities),
            }
        )
    return pd.DataFrame(rows)


def interval_rows(model, scores):
    rows = []
    for metric in ["average_precision", "roc_auc", "brier_score"]:
        rows.append(
            {
                "model": model,
                "metric": metric,
                "estimate": scores[metric].mean(),
                "lower_95": scores[metric].quantile(0.025),
                "upper_95": scores[metric].quantile(0.975),
                "bootstrap_repetitions": len(scores),
            }
        )
    return rows


def paired_difference(left, right, metric, repetitions, seed):
    merged = left.merge(
        right,
        on=["PlayerKey", "GameID", "split", "injury"],
        suffixes=("_with_surface", "_without_surface"),
        validate="one_to_one",
    )
    players = merged["PlayerKey"].drop_duplicates().to_numpy()
    generator = np.random.default_rng(seed)
    differences = []
    for _ in range(repetitions):
        sample = sampled_rows(merged, generator.choice(players, size=len(players), replace=True))
        labels = sample["injury"].to_numpy()
        if np.unique(labels).size < 2:
            continue
        with_surface = sample["probability_with_surface"].to_numpy()
        without_surface = sample["probability_without_surface"].to_numpy()
        differences.append(metric(labels, with_surface) - metric(labels, without_surface))
    values = np.asarray(differences)
    return {
        "mean_difference": float(values.mean()),
        "lower_95": float(np.quantile(values, 0.025)),
        "upper_95": float(np.quantile(values, 0.975)),
        "bootstrap_repetitions": int(len(values)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=340)
    args = parser.parse_args()

    prediction_files = sorted(args.model_dir.glob("*_predictions.csv"))
    if not prediction_files:
        raise FileNotFoundError(f"No prediction files found in {args.model_dir}")

    frames = {}
    intervals = []
    calibration_rows = []
    for path in prediction_files:
        model = path.name.removesuffix("_predictions.csv")
        frame = pd.read_csv(path)
        frames[model] = frame
        scores = bootstrap_scores(frame, args.repetitions, args.seed)
        intervals.extend(interval_rows(model, scores))
        observed, predicted = calibration_curve(
            frame["injury"],
            frame["probability"],
            n_bins=10,
            strategy="quantile",
        )
        calibration_rows.extend(
            {
                "model": model,
                "bin": index + 1,
                "mean_predicted_probability": float(predicted_value),
                "observed_fraction": float(observed_value),
            }
            for index, (predicted_value, observed_value) in enumerate(zip(predicted, observed))
        )

    pd.DataFrame(intervals).to_csv(args.model_dir / "bootstrap_intervals.csv", index=False)
    pd.DataFrame(calibration_rows).to_csv(args.model_dir / "calibration.csv", index=False)

    pairs = [
        ("full_logistic", "full_logistic_no_surface"),
        ("full_random_forest", "full_random_forest_no_surface"),
        ("full_xgboost", "full_xgboost_no_surface"),
    ]
    comparisons = []
    metric_functions = {
        "average_precision": average_precision_score,
        "roc_auc": roc_auc_score,
        "brier_score": brier_score_loss,
    }
    for with_surface, without_surface in pairs:
        if with_surface not in frames or without_surface not in frames:
            continue
        for metric_name, metric in metric_functions.items():
            difference = paired_difference(
                frames[with_surface],
                frames[without_surface],
                metric,
                args.repetitions,
                args.seed,
            )
            comparisons.append(
                {
                    "model_with_surface": with_surface,
                    "model_without_surface": without_surface,
                    "metric": metric_name,
                    **difference,
                }
            )
    pd.DataFrame(comparisons).to_csv(
        args.model_dir / "surface_increment_comparison.csv", index=False
    )

    metrics = pd.read_csv(args.model_dir / "metrics.csv")
    metrics = metrics.sort_values("average_precision", ascending=False)
    lines = [
        "# Model development summary",
        "",
        "| Model | Average precision | ROC-AUC | Precision | Recall | F1 | Brier |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in metrics.itertuples(index=False):
        lines.append(
            f"| {row.model} | {row.average_precision:.4f} | {row.roc_auc:.4f} | "
            f"{row.precision:.4f} | {row.recall:.4f} | {row.f1:.4f} | {row.brier_score:.4f} |"
        )
    lines.extend(
        [
            "",
            "Surface comparisons are paired cluster bootstraps over held-out players. A positive average-precision or ROC-AUC difference favors the model with surface. A negative Brier-score difference favors the model with surface.",
            "",
        ]
    )
    (args.model_dir / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"saved={args.model_dir} models={len(frames)} repetitions={args.repetitions}")


if __name__ == "__main__":
    main()
