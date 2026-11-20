import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


def verify(root):
    manifest = json.loads((root / "manifest.json").read_text())
    if "finished_utc" not in manifest:
        raise RuntimeError("Run is not complete: " + str(root))
    table = Path(manifest["settings"]["table"])
    if hashlib.sha256(table.read_bytes()).hexdigest() != manifest["table_sha256"]:
        raise RuntimeError("Input table changed after the run")
    for name, expected_hash in [
        ("run_feature_ablation.py", manifest["script_sha256"]),
        ("train_models.py", manifest["shared_model_code_sha256"]),
    ]:
        if (
            hashlib.sha256((root / "source" / name).read_bytes()).hexdigest()
            != expected_hash
        ):
            raise RuntimeError(
                "Saved code does not match the hash recorded for this run"
            )
    source = pd.read_csv(table)
    validation_players = set(source.loc[source.split.eq("validation"), "PlayerKey"])
    allowed_players = set(source.loc[source.split.isin(["train", "test"]), "PlayerKey"])
    folds = pd.read_csv(root / "fold_assignments.csv")
    metrics = pd.read_csv(root / "repeat_metrics.csv")
    fold_metrics = pd.read_csv(root / "fold_metrics.csv")
    settings = manifest["settings"]
    if (
        len(metrics) != settings["repeats"] * 12
        or len(fold_metrics) != settings["repeats"] * settings["folds"] * 12
    ):
        raise RuntimeError("Unexpected number of model evaluations")
    if folds.duplicated(["repeat", "row_id"]).any():
        raise RuntimeError("A row has multiple evaluation folds within a repeat")
    if folds.groupby(["repeat", "PlayerKey"]).fold.nunique().max() != 1:
        raise RuntimeError("A player is assigned to multiple folds")
    if (
        set(folds.PlayerKey) & validation_players
        or not set(folds.PlayerKey) <= allowed_players
    ):
        raise RuntimeError("Invalid player population")
    expected = folds.set_index(["repeat", "row_id"]).fold.sort_index()
    count = 0
    for path in sorted((root / "predictions").glob("*.csv")):
        frame = pd.read_csv(path)
        repeat = int(frame["repeat"].iloc[0])
        model = frame.model.iloc[0]
        feature_set = frame.feature_set.iloc[0]
        assignment = frame.set_index(["repeat", "row_id"]).fold.sort_index()
        if not assignment.equals(expected.loc[[repeat]]):
            raise RuntimeError("Inconsistent folds in " + path.name)
        if set(frame.PlayerKey) & validation_players:
            raise RuntimeError("Validation players in predictions")
        if (
            not np.isfinite(frame.probability).all()
            or not frame.probability.between(0, 1).all()
        ):
            raise RuntimeError("Invalid probabilities")
        observed = {
            "average_precision": average_precision_score(
                frame.injury, frame.probability
            ),
            "roc_auc": roc_auc_score(frame.injury, frame.probability),
            "brier_score": brier_score_loss(frame.injury, frame.probability),
        }
        saved = metrics[
            metrics["repeat"].eq(repeat)
            & metrics.model.eq(model)
            & metrics.feature_set.eq(feature_set)
        ].iloc[0]
        for metric, value in observed.items():
            if not np.isclose(value, saved[metric], rtol=1e-9, atol=1e-12):
                raise RuntimeError("Metric mismatch: " + path.name)
        count += 1
    if count != settings["repeats"] * 12:
        raise RuntimeError("Missing prediction files")
    features = manifest["feature_sets"]
    if set(features["context_surface"]) - set(features["context"]) != {"FieldType"}:
        raise RuntimeError("Invalid surface comparison")
    if set(features["context_history_surface"]) - set(features["context_history"]) != {
        "FieldType"
    }:
        raise RuntimeError("Invalid surface-with-history comparison")
    result = {
        "complete": True,
        "fits": len(fold_metrics),
        "verified_prediction_files": count,
        "metrics_recomputed": True,
        "identical_folds_across_configurations": True,
        "players_grouped": True,
        "validation_players_used": False,
        "input_hash_verified": True,
    }
    (root / "verification.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--primary", type=Path, required=True)
    parser.add_argument("--sensitivity", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    lines = [
        "# Workload and surface comparison",
        "",
        "We compare four input groups using logistic regression, random "
        "forest, and XGBoost. These are follow-up development results, not a "
        "new final validation test. Each run's manifest records the machine "
        "and device used.",
        "",
    ]
    for name, root in [
        ("Primary: exclude timing-conflicted players", args.primary),
        ("Sensitivity: retain players using listed game order", args.sensitivity),
    ]:
        checked = verify(root)
        audit = json.loads((root / "data_audit.json").read_text())
        metrics = pd.read_csv(root / "repeat_metrics.csv")
        paired = pd.read_csv(root / "paired_repeat_differences.csv")
        bootstrap = pd.read_csv(root / "player_bootstrap_differences.csv")
        lines += [
            "## " + name,
            "",
            f"Sample: {audit['development_players']} players, {audit['development_rows']} player-games, {audit['injury_games']} injury games. AP prevalence reference: {audit['prevalence']:.5f}.",
            "",
            "| Model | Feature group | Mean AP | Mean ROC-AUC | Mean Brier |",
            "|---|---|---:|---:|---:|",
        ]
        for (model, group), rows in metrics.groupby(
            ["model", "feature_set"], sort=False
        ):
            lines.append(
                f"| {model} | {group} | {rows.average_precision.mean():.5f} | {rows.roc_auc.mean():.5f} | {rows.brier_score.mean():.5f} |"
            )
        lines += [
            "",
            "AP changes compare each added input group with its matching "
            "baseline. For the intervals, we average predictions across "
            "repeats, then bootstrap players. These are not intervals for the "
            "difference between the mean repeat scores.",
            "",
            "| Model | Comparison | Mean repeat AP change | Bootstrap AP change | Lower 95% | Upper 95% |",
            "|---|---|---:|---:|---:|---:|",
        ]
        for row in bootstrap[bootstrap.metric.eq("average_precision")].itertuples():
            change = paired.loc[
                paired.model.eq(row.model) & paired.comparison.eq(row.comparison),
                "average_precision",
            ].mean()
            lines.append(
                f"| {row.model} | {row.comparison} | {change:+.5f} | {row.observed_difference:+.5f} | {row.lower_95:+.5f} | {row.upper_95:+.5f} |"
            )
        lines += [
            "",
            f"Checks passed: {checked['fits']} fits and {checked['verified_prediction_files']} prediction files. Models used the same folds, validation players were excluded, and scores recalculated from predictions matched the saved scores.",
            "",
        ]
        lines += [
            f"Predicting the sample injury rate for every game gives a Brier score of {audit['prevalence'] * (1 - audit['prevalence']):.5f}. Compare the model scores with this reference; lower is better. Ranking performance alone does not show that the probabilities are calibrated.",
            "",
        ]
    lines += [
        "## What the study contributes",
        "",
        "We test whether surface and earlier workload add useful information "
        "while keeping the models and folds the same. We also check the "
        "conflicting game dates and compare two ways of handling them. The "
        "contribution is this comparison and its findings, not a new "
        "algorithm or evidence of causation.",
        "",
        "## Reporting limits",
        "",
        "The reference study has 90 injury games in the primary sample and 93 "
        "in sensitivity. The target includes recorded lower-limb injuries, "
        "not verified non-contact injuries. Excluding players with "
        "conflicting dates changes the sample; retaining them assumes the "
        "listed game order is correct. Stadium and weather can still reveal "
        "surface information when FieldType is left out. We did not adjust "
        "for multiple comparisons or test an independent cohort. Because we "
        "had already seen the earlier validation results, this follow-up "
        "remains exploratory.",
        "",
        "The original final-validation random forest result remains a "
        "historical result for that earlier model only: AP 0.0527 and ROC-AUC "
        "0.7046. It is not the final score of the models evaluated in this "
        "follow-up.",
        "",
        "See docs/feature_ablation.md for the method. Each run also saves "
        "SUMMARY.md, manifest.json, data_audit.json, and verification.json "
        "with its settings and checks.",
    ]
    args.output.write_text("\n".join(lines) + "\n")
    print("saved=" + str(args.output))


if __name__ == "__main__":
    main()
