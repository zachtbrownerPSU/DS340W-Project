import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


SUBGROUP_COLUMNS = [
    "FieldType",
    "PositionGroup",
    "Weather_clean",
    "StadiumType_clean",
    "history_status",
    "prior_injury_status",
    "temperature_band",
]


def subgroup_metrics(frame, threshold):
    labels = frame["injury"].to_numpy()
    probabilities = frame["probability"].to_numpy()
    predictions = (probabilities >= threshold).astype(int)
    values = {
        "rows": len(frame),
        "players": frame["PlayerKey"].nunique(),
        "injuries": int(labels.sum()),
        "injury_rate": float(labels.mean()),
        "mean_probability": float(probabilities.mean()),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "brier_score": float(brier_score_loss(labels, probabilities)),
    }
    if np.unique(labels).size == 2:
        values["average_precision"] = float(
            average_precision_score(labels, probabilities)
        )
        values["roc_auc"] = float(roc_auc_score(labels, probabilities))
    else:
        values["average_precision"] = np.nan
        values["roc_auc"] = np.nan
    return values


def prepare_context(data):
    data = data.copy()
    data["history_status"] = np.where(
        data["prior_tracked_plays"].fillna(0).gt(0),
        "Prior tracked game",
        "No prior tracked game",
    )
    data["prior_injury_status"] = np.where(
        data["prior_injury_count"].fillna(0).gt(0),
        "Prior recorded injury",
        "No prior recorded injury",
    )
    data["temperature_band"] = pd.cut(
        data["Temperature"],
        bins=[-np.inf, 39, 59, 79, np.inf],
        labels=["Below 40", "40 to 59", "60 to 79", "80 or above"],
    ).astype("object")
    data["temperature_band"] = data["temperature_band"].fillna("Unknown")
    return data


def save_plot(rows, best_model, column, output_dir):
    selected = rows[
        rows["model"].eq(best_model) & rows["subgroup_variable"].eq(column)
    ].copy()
    selected = selected.sort_values("rows", ascending=False)
    if selected.empty:
        return
    labels = selected["subgroup"].astype(str).tolist()
    positions = np.arange(len(labels))
    width = 0.38
    figure, axis = plt.subplots(figsize=(max(7, len(labels) * 1.2), 4.8))
    axis.bar(
        positions - width / 2,
        selected["injury_rate"],
        width,
        color="white",
        edgecolor="black",
        label="Observed injury rate",
    )
    axis.bar(
        positions + width / 2,
        selected["mean_probability"],
        width,
        color="0.65",
        edgecolor="black",
        label="Mean predicted probability",
    )
    axis.set_xticks(positions)
    axis.set_xticklabels(labels, rotation=25, ha="right")
    axis.set_ylabel("Rate")
    axis.set_title(f"{best_model}: {column}")
    axis.legend(frameon=False)
    for spine in axis.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
    figure.tight_layout()
    figure.savefig(output_dir / f"{column}_rates.png", dpi=200, facecolor="white")
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--table", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--minimum-rows", type=int, default=25)
    args = parser.parse_args()

    data = prepare_context(pd.read_csv(args.table))
    data = data[data["split"].eq("test")].copy()
    metrics = pd.read_csv(args.model_dir / "metrics.csv")
    metrics = metrics[metrics["model"].ne("prevalence_baseline")]
    thresholds = metrics.set_index("model")["threshold"].to_dict()
    prediction_files = sorted(args.model_dir.glob("*_predictions.csv"))
    if not prediction_files:
        raise FileNotFoundError(f"No prediction files found in {args.model_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    error_rows = []

    context_columns = [
        "PlayerKey",
        "GameID",
        "FieldType",
        "PositionGroup",
        "Weather_clean",
        "StadiumType_clean",
        "history_status",
        "prior_injury_status",
        "temperature_band",
    ]
    for path in prediction_files:
        model = path.name.removesuffix("_predictions.csv")
        predictions = pd.read_csv(path)
        combined = predictions.merge(
            data[context_columns],
            on=["PlayerKey", "GameID"],
            how="left",
            validate="one_to_one",
        )
        threshold = float(thresholds[model])
        for column in SUBGROUP_COLUMNS:
            for subgroup, frame in combined.groupby(column, dropna=False):
                if len(frame) < args.minimum_rows:
                    continue
                rows.append(
                    {
                        "model": model,
                        "subgroup_variable": column,
                        "subgroup": str(subgroup),
                        **subgroup_metrics(frame, threshold),
                    }
                )
        false_positives = combined[combined["injury"].eq(0)].nlargest(
            25, "probability"
        )
        false_negatives = combined[combined["injury"].eq(1)].nsmallest(
            25, "probability"
        )
        for error_type, frame in [
            ("highest_scored_noninjury", false_positives),
            ("lowest_scored_injury", false_negatives),
        ]:
            frame = frame.copy()
            frame.insert(0, "error_type", error_type)
            frame.insert(0, "model", model)
            error_rows.append(frame)

    subgroup_results = pd.DataFrame(rows)
    subgroup_results.to_csv(args.output_dir / "subgroup_metrics.csv", index=False)
    pd.concat(error_rows, ignore_index=True).to_csv(
        args.output_dir / "largest_errors.csv", index=False
    )
    best_model = metrics.sort_values("average_precision", ascending=False).iloc[0]["model"]
    for column in SUBGROUP_COLUMNS:
        save_plot(subgroup_results, best_model, column, args.output_dir)
    print(
        f"saved={args.output_dir} best_model={best_model} "
        f"subgroup_rows={len(subgroup_results)} validation_used=false"
    )


if __name__ == "__main__":
    main()
