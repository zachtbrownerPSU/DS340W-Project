import argparse
import hashlib
import json
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import xgboost
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from train_models import (
    CONTEXT_CATEGORICAL,
    CONTEXT_NUMERIC,
    HISTORY_NUMERIC,
    build_preprocessor,
)

FEATURE_SETS = [
    "context",
    "context_surface",
    "context_history",
    "context_history_surface",
]
MODELS = ["logistic", "random_forest", "xgboost"]
METRICS = ["average_precision", "roc_auc", "brier_score"]
PAIRS = [
    ("context_surface", "context", "surface_without_history"),
    ("context_history_surface", "context_history", "surface_with_history"),
    ("context_history", "context", "history_without_surface"),
    ("context_history_surface", "context_surface", "history_with_surface"),
]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def gpu_inventory(device):
    executable = shutil.which("nvidia-smi")
    if executable is None:
        if device != "cpu":
            raise RuntimeError("nvidia-smi is unavailable; use --device cpu or check the NVIDIA installation")
        return "Unavailable; CPU execution selected"
    result = subprocess.run(
        [executable, "-L"], capture_output=True, text=True, timeout=30
    )
    if result.returncode != 0:
        if device != "cpu":
            raise RuntimeError("NVIDIA inventory failed: " + result.stderr.strip())
        return "Unavailable; CPU execution selected"
    return result.stdout


def scores(labels, probabilities):
    return {
        "average_precision": float(average_precision_score(labels, probabilities)),
        "roc_auc": float(roc_auc_score(labels, probabilities)),
        "brier_score": float(brier_score_loss(labels, probabilities)),
    }


def audit_data(data, allow_conflicts=False):
    if data[["PlayerKey", "GameID"]].duplicated().any():
        raise ValueError("Duplicate player-game keys")
    if not data["injury"].isin([0, 1]).all():
        raise ValueError("Invalid injury labels")
    if data.groupby("PlayerKey")["split"].nunique().max() != 1:
        raise ValueError("Player appears in multiple original splits")
    ordered = data.sort_values(["PlayerKey", "PlayerGame", "PlayerDay", "GameID"])
    chronology_conflicts = (
        ordered.groupby("PlayerKey")["PlayerDay"].diff().dropna().le(0)
    )
    if chronology_conflicts.any() and not allow_conflicts:
        raise ValueError("Game chronology is not strictly increasing within a player")
    checks = {}
    for feature in HISTORY_NUMERIC:
        rolling = feature.startswith("prior_3_")
        suffix = feature[len("prior_3_") :] if rolling else feature[len("prior_") :]
        previous = ordered.groupby("PlayerKey")["game_" + suffix].shift(1)
        if rolling:
            previous = previous.groupby(ordered["PlayerKey"]).transform(
                lambda x: x.rolling(3, min_periods=1).mean()
            )
        matched = np.allclose(
            previous.to_numpy(float), ordered[feature].to_numpy(float), equal_nan=True
        )
        checks[feature] = bool(matched)
        if not matched:
            raise ValueError("History audit failed for " + feature)
    prior_injuries = ordered.groupby("PlayerKey")["injury"].cumsum() - ordered["injury"]
    if not np.array_equal(prior_injuries, ordered["prior_injury_count"]):
        raise ValueError("Prior injury count includes incorrect events")
    if not np.allclose(
        ordered.groupby("PlayerKey")["PlayerDay"].diff(),
        ordered["days_since_previous_game"],
        equal_nan=True,
    ):
        raise ValueError("Incorrect time since previous game")
    if not np.array_equal(
        ordered.groupby("PlayerKey").cumcount(), ordered["games_before"]
    ):
        raise ValueError("Incorrect prior game count")
    allowed = set(CONTEXT_CATEGORICAL + CONTEXT_NUMERIC + HISTORY_NUMERIC)
    forbidden = {
        "PlayerKey",
        "GameID",
        "injury",
        "DM_M1",
        "DM_M7",
        "DM_M28",
        "DM_M42",
        "split",
    }
    if allowed & forbidden or any(x.startswith("game_") for x in allowed):
        raise ValueError("Current-game outcomes or workload in model inputs")
    return {
        "development_rows": len(data),
        "development_players": int(data.PlayerKey.nunique()),
        "injury_games": int(data.injury.sum()),
        "prevalence": float(data.injury.mean()),
        "history_checks": checks,
        "chronology_checked": True,
        "chronology_conflicting_intervals": int(chronology_conflicts.sum()),
        "prior_injury_count_checked": True,
        "current_game_workload_excluded": True,
        "player_identifiers_excluded": True,
        "games_without_tracking": int(data.game_tracked_plays.eq(0).sum()),
        "games_with_partial_tracking": int(
            (
                (data.game_tracked_plays > 0)
                & (data.game_tracked_plays < data.game_plays)
            ).sum()
        ),
        "rows_without_previous_game": int(data.games_before.eq(0).sum()),
        "rows_with_previous_game_but_no_prior_tracking": int(
            (data.games_before.gt(0) & data.prior_tracked_plays.eq(0)).sum()
        ),
    }


def feature_columns(name):
    categorical = [x for x in CONTEXT_CATEGORICAL if x != "FieldType"]
    numeric = list(CONTEXT_NUMERIC)
    if name.endswith("surface"):
        categorical.append("FieldType")
    if "history" in name:
        numeric += HISTORY_NUMERIC
    return categorical, numeric


def make_pipeline(model, feature_set, labels, seed, threads, device):
    categorical, numeric = feature_columns(feature_set)
    if model == "logistic":
        estimator = LogisticRegression(
            class_weight="balanced",
            max_iter=3000,
            solver="liblinear",
            random_state=seed,
        )
    elif model == "random_forest":
        estimator = RandomForestClassifier(
            n_estimators=600,
            min_samples_leaf=3,
            max_features="sqrt",
            class_weight="balanced_subsample",
            n_jobs=threads,
            random_state=seed,
        )
    else:
        estimator = XGBClassifier(
            n_estimators=700,
            max_depth=4,
            learning_rate=0.03,
            subsample=0.85,
            colsample_bytree=0.85,
            min_child_weight=3,
            reg_lambda=1.0,
            scale_pos_weight=float((labels == 0).sum() / labels.sum()),
            eval_metric="logloss",
            tree_method="hist",
            device=device,
            n_jobs=threads,
            random_state=seed,
        )
    return (
        Pipeline(
            [
                ("preprocess", build_preprocessor(categorical, numeric)),
                ("model", estimator),
            ]
        ),
        categorical + numeric,
    )


def bootstrap_comparisons(data, predictions, repetitions, seed):
    averaged = predictions.groupby(
        ["model", "feature_set", "row_id"], as_index=False
    ).probability.mean()
    players = data.PlayerKey.unique()
    player_rows = {p: np.flatnonzero(data.PlayerKey.to_numpy() == p) for p in players}
    probability_map = {}
    for (model, feature_set), group in averaged.groupby(["model", "feature_set"]):
        probability_map[(model, feature_set)] = (
            group.set_index("row_id")
            .probability.reindex(np.arange(len(data)))
            .to_numpy()
        )
    labels = data.injury.to_numpy()
    differences = {
        (model, label, metric): []
        for model in MODELS
        for _, _, label in PAIRS
        for metric in METRICS
    }
    rng = np.random.default_rng(seed)
    valid = 0
    for iteration in range(repetitions):
        sampled = rng.choice(players, size=len(players), replace=True)
        indices = np.concatenate([player_rows[p] for p in sampled])
        y = labels[indices]
        if np.unique(y).size < 2:
            continue
        valid += 1
        values = {
            (model, feature_set): scores(
                y, probability_map[(model, feature_set)][indices]
            )
            for model in MODELS
            for feature_set in FEATURE_SETS
        }
        for model in MODELS:
            for with_set, without_set, label in PAIRS:
                for metric in METRICS:
                    differences[(model, label, metric)].append(
                        values[(model, with_set)][metric]
                        - values[(model, without_set)][metric]
                    )
    rows = []
    for model in MODELS:
        for with_set, without_set, label in PAIRS:
            with_scores = scores(labels, probability_map[(model, with_set)])
            without_scores = scores(labels, probability_map[(model, without_set)])
            for metric in METRICS:
                values = np.asarray(differences[(model, label, metric)])
                rows.append(
                    {
                        "model": model,
                        "comparison": label,
                        "metric": metric,
                        "observed_difference": with_scores[metric]
                        - without_scores[metric],
                        "lower_95": float(np.quantile(values, 0.025)),
                        "upper_95": float(np.quantile(values, 0.975)),
                        "valid_bootstrap_draws": valid,
                    }
                )
    return pd.DataFrame(rows), probability_map


def draw_curves(data, probability_map, output):
    plt.rcParams.update(
        {"axes.spines.top": True, "axes.spines.right": True, "font.size": 10}
    )
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    styles = ["-", "--", "-.", ":"]
    for column, model in enumerate(MODELS):
        for name, style in zip(FEATURE_SETS, styles):
            probabilities = probability_map[(model, name)]
            fpr, tpr, _ = roc_curve(data.injury, probabilities)
            precision, recall, _ = precision_recall_curve(data.injury, probabilities)
            axes[0, column].plot(fpr, tpr, linestyle=style, color="black", label=name)
            axes[1, column].plot(
                recall, precision, linestyle=style, color="black", label=name
            )
        axes[0, column].plot([0, 1], [0, 1], color="gray", linewidth=0.7)
        axes[1, column].axhline(data.injury.mean(), color="gray", linewidth=0.7)
        axes[0, column].set(
            title=model,
            xlabel="False positive rate",
            ylabel="True positive rate",
            xlim=(0, 1),
            ylim=(0, 1),
        )
        axes[1, column].set(
            xlabel="Recall", ylabel="Precision", xlim=(0, 1), ylim=(0, 1)
        )
    axes[1, 2].legend(fontsize=8)
    fig.suptitle("Exploratory development predictions averaged over grouped CV repeats")
    fig.tight_layout()
    fig.savefig(output / "roc_precision_recall.png", dpi=180)
    fig.savefig(output / "roc_precision_recall.pdf")
    plt.close(fig)


def write_summary(audit, pooled, paired, bootstrap, args):
    lines = [
        "# Feature ablation results",
        "",
        "## Scope",
        "",
        "Exploratory follow-up after the original final validation results were examined. Only original train and test players were fitted or scored. No new final validation claim is made.",
        "",
        f"Development sample: {audit['development_rows']} player-games, {audit['development_players']} players, {audit['injury_games']} injury games. Prevalence: {audit['prevalence']:.4%}.",
        "",
        f"All 12 configurations used the same {args.repeats} repetitions of {args.folds} player-grouped folds. Hyperparameters were fixed before this run. Class weights for XGBoost were calculated within each training fold. Imputation, scaling, and encoding were fitted within each training fold.",
        "",
        "## Feature groups",
        "",
        "Context includes position, stadium, weather, temperature, player-day, player-game, earlier game count, days since the previous game, and prior injury count. History adds previous-game and prior-three-game workload and movement summaries. Surface is FieldType only; stadium and weather can still proxy surface, so this isolates the extra value of explicitly providing surface.",
        "",
        "## Performance",
        "",
        "The table reports the mean of complete out-of-fold metrics over repeats, not the mean of fold-level AP values. Fold percentiles and repeat variation are descriptive and are not independent-sample confidence intervals.",
        "",
        "| Model | Feature set | Mean AP | Mean ROC-AUC | Mean Brier |",
        "|---|---|---:|---:|---:|",
    ]
    for (model, feature_set), group in pooled.groupby(
        ["model", "feature_set"], sort=False
    ):
        lines.append(
            f"| {model} | {feature_set} | {group.average_precision.mean():.5f} | {group.roc_auc.mean():.5f} | {group.brier_score.mean():.5f} |"
        )
    lines += [
        "",
        f"Constant-risk AP reference: {audit['prevalence']:.5f}. Lower Brier is better; higher AP and ROC-AUC are better.",
        "",
        "## Paired changes",
        "",
        "Differences are the added-feature configuration minus the otherwise identical configuration. Fractions describe repeats and are not p-values.",
        "",
        "| Model | Added features | Mean AP change | Repeats with AP improvement |",
        "|---|---|---:|---:|",
    ]
    for (model, comparison), group in paired.groupby(
        ["model", "comparison"], sort=False
    ):
        lines.append(
            f"| {model} | {comparison} | {group.average_precision.mean():+.5f} | {(group.average_precision > 0).sum()}/{len(group)} |"
        )
    lines += [
        "",
        "## Player bootstrap",
        "",
        "These intervals resample players with replacement using predictions averaged across CV repeats. Each player's predictions were made by models excluding that player. The intervals condition on the fitted models, omit model-refitting uncertainty, and describe an averaged cross-validation predictor rather than a separately tested deployed model. Multiple comparisons are exploratory and unadjusted.",
        "",
        "| Model | Comparison | Observed AP change | Lower 95% | Upper 95% |",
        "|---|---|---:|---:|---:|",
    ]
    for row in bootstrap[bootstrap.metric.eq("average_precision")].itertuples():
        lines.append(
            f"| {row.model} | {row.comparison} | {row.observed_difference:+.5f} | {row.lower_95:+.5f} | {row.upper_95:+.5f} |"
        )
    lines += [
        "",
        "## Timing and data audit",
        "",
        "All 18 history columns were independently recomputed using the listed PlayerGame order and matched the saved table. Earlier injury counts, days since prior game, and earlier game counts matched that order. Current-game workload, injury severity, identifiers, and current injury outcomes were excluded from inputs.",
        "",
        f"Chronology policy: {args.chronology_policy}. The original development data had {audit['original_conflicting_players']} players with a backward or repeated PlayerDay interval. The primary exclude_players policy removes all games for those players before splitting folds. The game_order sensitivity retains them, assumes PlayerGame defines the true order, and replaces nonpositive rest intervals with missing values. That assumption does not resolve the underlying date inconsistency. See chronology_conflicts.csv for affected player-games.",
        "",
        f"There were {audit['games_without_tracking']} games without tracking, {audit['games_with_partial_tracking']} with partial tracking, {audit['rows_without_previous_game']} with no previous game, and {audit['rows_with_previous_game_but_no_prior_tracking']} with a previous game but no prior tracking. Missing tracking and zero totals can influence predictions; history therefore combines measured movement, exposure, and data-availability signals. This experiment does not isolate their individual effects.",
        "",
        "## Limits and interpretation",
        "",
        "The target is whether a player-game contains a recorded lower-limb injury. The code does not establish non-contact mechanism or predict a diagnosis. The cohort is small and selected; unrecorded injuries, confounding, and the limited generalizability of random player splits remain concerns. PlayerDay is relative time, so this is not a calendar-based future-season test. A change in predictive performance is not a causal effect of turf or workload. Bootstrap intervals crossing zero indicate uncertainty, and neither a positive point estimate nor a null result is proof of a general effect.",
        "",
        "The earlier validation sample was already viewed before this study. An independent cohort would be needed to confirm changes selected using these results. Larger compute capacity does not increase the number of independent injury events.",
        "",
        "## Reproduce",
        "",
        "```bash",
        f"python scripts/run_feature_ablation.py --table {args.table} --output-dir outputs/feature_ablation_new_run --repeats {args.repeats} --folds {args.folds} --bootstrap {args.bootstrap} --seed {args.seed} --threads {args.threads} --device {args.device} --chronology-policy {args.chronology_policy}",
        "```",
        "",
        "See manifest.json for hashes, features, settings, package versions, and server information. See fold_assignments.csv, fold_metrics.csv, repeat_metrics.csv, paired_repeat_differences.csv, player_bootstrap_differences.csv, and predictions/ for the complete results.",
    ]
    (args.output_dir / "SUMMARY.md").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--table", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=340)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument(
        "--device", choices=["cpu", "cuda:0", "cuda:1"], default="cuda:0"
    )
    parser.add_argument(
        "--chronology-policy",
        choices=["exclude_players", "game_order"],
        default="exclude_players",
    )
    args = parser.parse_args()
    if args.output_dir.exists():
        raise ValueError("Output directory already exists; choose a new directory")
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "predictions").mkdir()
    (args.output_dir / "source").mkdir()
    for source in [Path(__file__), Path(__file__).with_name("train_models.py")]:
        (args.output_dir / "source" / source.name).write_bytes(source.read_bytes())
    data = pd.read_csv(args.table)
    data = data.loc[data.split.isin(["train", "test"])].reset_index(drop=True)
    ordered = data.sort_values(["PlayerKey", "PlayerGame", "PlayerDay", "GameID"])
    conflicting = ordered.loc[
        ordered.groupby("PlayerKey").PlayerDay.diff().le(0), "PlayerKey"
    ].unique()
    affected = data[data.PlayerKey.isin(conflicting)]
    affected.to_csv(args.output_dir / "chronology_conflicts.csv", index=False)
    if args.chronology_policy == "exclude_players":
        data = data.loc[~data.PlayerKey.isin(conflicting)].reset_index(drop=True)
    audit = audit_data(data, allow_conflicts=args.chronology_policy == "game_order")
    audit["original_conflicting_players"] = len(conflicting)
    audit["original_conflicting_player_games"] = len(affected)
    audit["original_conflicting_injury_games"] = int(affected.injury.sum())
    if args.chronology_policy == "game_order":
        data.loc[data.days_since_previous_game.le(0), "days_since_previous_game"] = (
            np.nan
        )
    (args.output_dir / "data_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    manifest = {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": platform.node(),
        "python": sys.version,
        "packages": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "sklearn": sklearn.__version__,
            "xgboost": xgboost.__version__,
        },
        "table_sha256": sha256(args.table),
        "script_sha256": sha256(Path(__file__)),
        "shared_model_code_sha256": sha256(Path(__file__).with_name("train_models.py")),
        "settings": {
            k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
        },
        "feature_sets": {name: sum(feature_columns(name), []) for name in FEATURE_SETS},
        "model_parameters": {
            name: make_pipeline(
                name,
                "context",
                data.injury.to_numpy(),
                args.seed,
                args.threads,
                args.device,
            )[0]
            .named_steps["model"]
            .get_params()
            for name in MODELS
        },
        "original_validation_previously_examined": True,
        "validation_players_used": False,
        "target": "recorded lower-limb injury during a player-game",
    }
    manifest["gpu_inventory"] = gpu_inventory(args.device)
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    assignments = []
    fold_rows = []
    repeat_rows = []
    prediction_frames = []
    labels = data.injury.to_numpy()
    for repeat in range(args.repeats):
        splitter = StratifiedGroupKFold(
            n_splits=args.folds, shuffle=True, random_state=args.seed + repeat
        )
        folds = list(splitter.split(data, labels, data.PlayerKey))
        for fold, (training, evaluation) in enumerate(folds, start=1):
            if set(data.iloc[training].PlayerKey) & set(
                data.iloc[evaluation].PlayerKey
            ):
                raise ValueError("Player overlap across a fold")
            if (
                np.unique(labels[evaluation]).size != 2
                or np.unique(labels[training]).size != 2
            ):
                raise ValueError("A fold lacks both outcome classes")
            for row_id in evaluation:
                assignments.append(
                    {
                        "repeat": repeat + 1,
                        "fold": fold,
                        "row_id": int(row_id),
                        "PlayerKey": int(data.iloc[row_id].PlayerKey),
                        "GameID": data.iloc[row_id].GameID,
                    }
                )
        pd.DataFrame(assignments).to_csv(
            args.output_dir / "fold_assignments.csv", index=False
        )
        for model in MODELS:
            for feature_set in FEATURE_SETS:
                probabilities = np.full(len(data), np.nan)
                fold_numbers = np.zeros(len(data), dtype=int)
                for fold, (training, evaluation) in enumerate(folds, start=1):
                    pipeline, features = make_pipeline(
                        model,
                        feature_set,
                        labels[training],
                        args.seed + repeat,
                        args.threads,
                        args.device,
                    )
                    pipeline.fit(data.iloc[training][features], labels[training])
                    if model == "xgboost":
                        pipeline.named_steps["model"].set_params(device="cpu")
                    probabilities[evaluation] = pipeline.predict_proba(
                        data.iloc[evaluation][features]
                    )[:, 1]
                    fold_numbers[evaluation] = fold
                    fold_rows.append(
                        {
                            "repeat": repeat + 1,
                            "fold": fold,
                            "model": model,
                            "feature_set": feature_set,
                            "evaluation_rows": len(evaluation),
                            "injuries": int(labels[evaluation].sum()),
                            **scores(labels[evaluation], probabilities[evaluation]),
                        }
                    )
                    print(
                        f"repeat={repeat + 1} fold={fold} model={model} features={feature_set} ap={fold_rows[-1]['average_precision']:.5f}",
                        flush=True,
                    )
                if np.isnan(probabilities).any():
                    raise ValueError("Missing out-of-fold predictions")
                frame = data[["PlayerKey", "GameID", "injury"]].copy()
                frame.insert(0, "row_id", np.arange(len(data)))
                frame["repeat"] = repeat + 1
                frame["fold"] = fold_numbers
                frame["model"] = model
                frame["feature_set"] = feature_set
                frame["probability"] = probabilities
                frame.to_csv(
                    args.output_dir
                    / "predictions"
                    / f"repeat_{repeat + 1}_{model}_{feature_set}.csv",
                    index=False,
                )
                prediction_frames.append(frame)
                repeat_rows.append(
                    {
                        "repeat": repeat + 1,
                        "model": model,
                        "feature_set": feature_set,
                        **scores(labels, probabilities),
                    }
                )
                pd.DataFrame(fold_rows).to_csv(
                    args.output_dir / "fold_metrics.csv", index=False
                )
                pd.DataFrame(repeat_rows).to_csv(
                    args.output_dir / "repeat_metrics.csv", index=False
                )
    pooled = pd.DataFrame(repeat_rows)
    paired_rows = []
    for repeat in range(1, args.repeats + 1):
        for model in MODELS:
            subset = pooled[
                pooled.repeat.eq(repeat) & pooled.model.eq(model)
            ].set_index("feature_set")
            for with_set, without_set, comparison in PAIRS:
                paired_rows.append(
                    {
                        "repeat": repeat,
                        "model": model,
                        "comparison": comparison,
                        **{
                            metric: float(
                                subset.loc[with_set, metric]
                                - subset.loc[without_set, metric]
                            )
                            for metric in METRICS
                        },
                    }
                )
    paired = pd.DataFrame(paired_rows)
    paired.to_csv(args.output_dir / "paired_repeat_differences.csv", index=False)
    bootstrap, probability_map = bootstrap_comparisons(
        data, pd.concat(prediction_frames, ignore_index=True), args.bootstrap, args.seed
    )
    bootstrap.to_csv(args.output_dir / "player_bootstrap_differences.csv", index=False)
    draw_curves(data, probability_map, args.output_dir)
    write_summary(audit, pooled, paired, bootstrap, args)
    manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(f"completed={args.output_dir} validation_players_used=false", flush=True)


if __name__ == "__main__":
    main()
