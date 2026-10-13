import argparse
from pathlib import Path

import pandas as pd


def number(value, digits=4):
    if pd.isna(value):
        return "NA"
    return f"{value:.{digits}f}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs-dir", type=Path, required=True)
    parser.add_argument("--output-file", type=Path, required=True)
    args = parser.parse_args()
    lines = [
        "# Extended analysis summary",
        "",
        "All results below use training and testing players only. Validation players were not evaluated.",
        "",
    ]

    exposure = pd.read_csv(args.outputs_dir / "surface_inference/surface_exposure.csv")
    effects = pd.read_csv(
        args.outputs_dir / "surface_inference/surface_effect_estimates.csv"
    )
    within = pd.read_csv(
        args.outputs_dir / "surface_inference/within_player_surface_difference.csv"
    ).iloc[0]
    lines.extend(
        [
            "## Surface results",
            "",
            "| Surface | Player-games | Injuries | Injuries per 1,000 player-games |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for row in exposure.itertuples(index=False):
        lines.append(
            f"| {row.surface} | {row.player_games} | {row.injuries} | "
            f"{row.injuries_per_1000_player_games:.3f} |"
        )
    lines.extend(
        [
            "",
            "| Measure | Bootstrap estimate | Lower 95% | Upper 95% |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for row in effects.itertuples(index=False):
        lines.append(
            f"| {row.metric} | {number(row.estimate)} | {number(row.lower_95)} | "
            f"{number(row.upper_95)} |"
        )
    lines.extend(
        [
            "",
            f"The within-player mean risk difference was {number(within.point_estimate)} among "
            f"{int(within.dual_surface_players)} players observed on both surfaces. Its player-bootstrap "
            f"95% interval was {number(within.lower_95)} to {number(within.upper_95)}.",
            "",
        ]
    )

    repeated = pd.read_csv(args.outputs_dir / "repeated_group_cv/metric_summary.csv")
    repeated = repeated[repeated["metric"].eq("average_precision")].sort_values(
        "mean", ascending=False
    )
    lines.extend(
        [
            "## Repeated grouped cross-validation",
            "",
            "| Model | Mean average precision | Standard deviation | 2.5th percentile | 97.5th percentile |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in repeated.itertuples(index=False):
        lines.append(
            f"| {row.model} | {number(row.mean)} | {number(row.standard_deviation)} | "
            f"{number(row.lower_2_5)} | {number(row.upper_97_5)} |"
        )
    lines.append("")

    search = pd.read_csv(args.outputs_dir / "xgboost_search/search_results.csv")
    comparison = pd.read_csv(
        args.outputs_dir / "xgboost_search/best_model_oof_metrics.csv"
    )
    lines.extend(
        [
            "## XGBoost search",
            "",
            f"The search evaluated {len(search)} settings. The best trial had mean fold average precision "
            f"{number(search.iloc[0]['mean_average_precision'])}.",
            "",
            "| Model | Out-of-fold average precision | ROC-AUC | Brier score |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for row in comparison.itertuples(index=False):
        lines.append(
            f"| {row.model} | {number(row.average_precision)} | {number(row.roc_auc)} | "
            f"{number(row.brier_score)} |"
        )
    lines.append("")

    lines.extend(
        [
            "## Completed-play prediction",
            "",
            "| Window | Best model | Average precision | Precision | Recall | Injury-game alert rate | False alerts per 1,000 negative plays |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for window in [5, 10, 20]:
        metrics = pd.read_csv(
            args.outputs_dir
            / f"play_sequence_models/within_{window}_plays/metrics.csv"
        )
        row = metrics.sort_values("average_precision", ascending=False).iloc[0]
        lines.append(
            f"| {window} plays | {row['model']} | {number(row['average_precision'])} | "
            f"{number(row['precision'])} | {number(row['recall'])} | "
            f"{number(row['injury_game_alert_rate'])} | "
            f"{number(row['false_alerts_per_1000_negative_plays'], 2)} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation limits",
            "",
            "The dataset has few injuries, repeated rows from the same players, and no random assignment of playing surface. Cross-validation, bootstrapping, and within-player comparisons measure uncertainty but do not remove all confounding. The completed-play models are exploratory and use fewer timed injury events because injury records without a PlayKey are excluded.",
            "",
        ]
    )
    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    args.output_file.write_text("\n".join(lines), encoding="utf-8")
    print(f"saved={args.output_file} validation_used=false")


if __name__ == "__main__":
    main()
