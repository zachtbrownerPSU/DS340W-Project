import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact


SEVERITY_COLUMNS = ["DM_M1", "DM_M7", "DM_M28", "DM_M42"]


def canonical_surface(value):
    text = str(value).strip().lower()
    if "synthetic" in text or "artificial" in text or "turf" in text:
        return "Synthetic"
    if "natural" in text or "grass" in text:
        return "Natural"
    return str(value).strip().title()


def interval(values):
    finite = values[np.isfinite(values)]
    return {
        "estimate": float(np.mean(finite)),
        "lower_95": float(np.quantile(finite, 0.025)),
        "upper_95": float(np.quantile(finite, 0.975)),
        "bootstrap_repetitions": int(len(finite)),
    }


def holm_adjust(values):
    values = np.asarray(values, dtype=float)
    order = np.argsort(values)
    adjusted = np.empty(len(values), dtype=float)
    running = 0.0
    count = len(values)
    for rank, index in enumerate(order):
        candidate = min(1.0, values[index] * (count - rank))
        running = max(running, candidate)
        adjusted[index] = running
    return adjusted


def player_surface_arrays(data, players):
    counts = (
        data.groupby(["PlayerKey", "surface"])
        .agg(exposures=("injury", "size"), injuries=("injury", "sum"))
        .unstack(fill_value=0)
    )
    counts = counts.reindex(players, fill_value=0)
    arrays = {}
    for surface in ["Natural", "Synthetic"]:
        for measure in ["exposures", "injuries"]:
            column = (measure, surface)
            if column in counts.columns:
                arrays[f"{surface}_{measure}"] = counts[column].to_numpy(dtype=float)
            else:
                arrays[f"{surface}_{measure}"] = np.zeros(len(players), dtype=float)
    return arrays


def bootstrap_surface(data, repetitions, seed):
    players = data["PlayerKey"].drop_duplicates().to_numpy()
    arrays = player_surface_arrays(data, players)
    generator = np.random.default_rng(seed)
    sampled = generator.integers(0, len(players), size=(repetitions, len(players)))
    natural_exposures = arrays["Natural_exposures"][sampled].sum(axis=1)
    natural_injuries = arrays["Natural_injuries"][sampled].sum(axis=1)
    synthetic_exposures = arrays["Synthetic_exposures"][sampled].sum(axis=1)
    synthetic_injuries = arrays["Synthetic_injuries"][sampled].sum(axis=1)
    natural_risk = natural_injuries / natural_exposures
    synthetic_risk = synthetic_injuries / synthetic_exposures
    result = pd.DataFrame(
        {
            "repetition": np.arange(1, repetitions + 1),
            "natural_risk": natural_risk,
            "synthetic_risk": synthetic_risk,
            "risk_difference": synthetic_risk - natural_risk,
            "risk_ratio": synthetic_risk / natural_risk,
            "odds_ratio": (
                synthetic_injuries
                * (natural_exposures - natural_injuries)
                / (
                    natural_injuries
                    * (synthetic_exposures - synthetic_injuries)
                )
            ),
        }
    )
    return result


def within_player_analysis(data, repetitions, seed):
    grouped = (
        data.groupby(["PlayerKey", "surface"])
        .agg(exposures=("injury", "size"), injuries=("injury", "sum"))
        .unstack(fill_value=0)
    )
    required = [
        ("exposures", "Natural"),
        ("exposures", "Synthetic"),
        ("injuries", "Natural"),
        ("injuries", "Synthetic"),
    ]
    for column in required:
        if column not in grouped.columns:
            grouped[column] = 0
    dual = grouped[
        (grouped[("exposures", "Natural")] > 0)
        & (grouped[("exposures", "Synthetic")] > 0)
    ].copy()
    differences = (
        dual[("injuries", "Synthetic")] / dual[("exposures", "Synthetic")]
        - dual[("injuries", "Natural")] / dual[("exposures", "Natural")]
    ).to_numpy()
    generator = np.random.default_rng(seed)
    sampled = generator.choice(
        differences, size=(repetitions, len(differences)), replace=True
    )
    bootstrap = sampled.mean(axis=1)
    values = interval(bootstrap)
    values["point_estimate"] = float(differences.mean())
    values["dual_surface_players"] = int(len(differences))
    return values, pd.DataFrame(
        {
            "PlayerKey": dual.index,
            "natural_exposures": dual[("exposures", "Natural")].to_numpy(),
            "natural_injuries": dual[("injuries", "Natural")].to_numpy(),
            "synthetic_exposures": dual[("exposures", "Synthetic")].to_numpy(),
            "synthetic_injuries": dual[("injuries", "Synthetic")].to_numpy(),
            "player_risk_difference": differences,
        }
    )


def severity_tests(data):
    injured = data[data["injury"].eq(1)]
    rows = []
    for column in SEVERITY_COLUMNS:
        synthetic = injured[injured["surface"].eq("Synthetic")][column]
        natural = injured[injured["surface"].eq("Natural")][column]
        table = np.array(
            [
                [int(synthetic.sum()), int(len(synthetic) - synthetic.sum())],
                [int(natural.sum()), int(len(natural) - natural.sum())],
            ]
        )
        odds_ratio, p_value = fisher_exact(table)
        rows.append(
            {
                "threshold": column,
                "synthetic_injuries": len(synthetic),
                "synthetic_at_or_above_threshold": int(synthetic.sum()),
                "synthetic_fraction": float(synthetic.mean()),
                "natural_injuries": len(natural),
                "natural_at_or_above_threshold": int(natural.sum()),
                "natural_fraction": float(natural.mean()),
                "odds_ratio": float(odds_ratio),
                "fisher_p_value": float(p_value),
            }
        )
    result = pd.DataFrame(rows)
    result["holm_adjusted_p_value"] = holm_adjust(result["fisher_p_value"])
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--table", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=340)
    args = parser.parse_args()

    data = pd.read_csv(args.table)
    data = data[data["split"].isin(["train", "test"])].copy()
    data["surface"] = data["FieldType"].map(canonical_surface)
    available = set(data["surface"].dropna().unique())
    if not {"Natural", "Synthetic"}.issubset(available):
        raise ValueError(f"Natural and Synthetic surfaces were not both found: {available}")
    data = data[data["surface"].isin(["Natural", "Synthetic"])].copy()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    exposure = (
        data.groupby("surface")
        .agg(
            player_games=("injury", "size"),
            injuries=("injury", "sum"),
            players=("PlayerKey", "nunique"),
        )
        .reset_index()
    )
    exposure["injuries_per_1000_player_games"] = (
        exposure["injuries"] * 1000 / exposure["player_games"]
    )
    exposure.to_csv(args.output_dir / "surface_exposure.csv", index=False)

    natural = data[data["surface"].eq("Natural")]["injury"]
    synthetic = data[data["surface"].eq("Synthetic")]["injury"]
    table = np.array(
        [
            [int(synthetic.sum()), int(len(synthetic) - synthetic.sum())],
            [int(natural.sum()), int(len(natural) - natural.sum())],
        ]
    )
    odds_ratio, p_value = fisher_exact(table)
    bootstrap = bootstrap_surface(data, args.repetitions, args.seed)
    bootstrap.to_csv(args.output_dir / "surface_cluster_bootstrap.csv", index=False)
    estimate_rows = []
    for metric in ["risk_difference", "risk_ratio", "odds_ratio"]:
        values = interval(bootstrap[metric].to_numpy())
        values["metric"] = metric
        estimate_rows.append(values)
    estimates = pd.DataFrame(estimate_rows)
    estimates["fisher_exact_odds_ratio"] = float(odds_ratio)
    estimates["fisher_exact_p_value"] = float(p_value)
    estimates.to_csv(args.output_dir / "surface_effect_estimates.csv", index=False)

    within_values, within_players = within_player_analysis(
        data, args.repetitions, args.seed
    )
    within_players.to_csv(args.output_dir / "dual_surface_players.csv", index=False)
    pd.DataFrame([within_values]).to_csv(
        args.output_dir / "within_player_surface_difference.csv", index=False
    )
    severity_tests(data).to_csv(
        args.output_dir / "injury_severity_by_surface.csv", index=False
    )
    position_rates = (
        data.groupby(["PositionGroup", "surface"])
        .agg(player_games=("injury", "size"), injuries=("injury", "sum"))
        .reset_index()
    )
    position_rates["injuries_per_1000_player_games"] = (
        position_rates["injuries"] * 1000 / position_rates["player_games"]
    )
    position_rates.to_csv(args.output_dir / "surface_position_rates.csv", index=False)
    print(
        f"saved={args.output_dir} player_games={len(data)} players={data['PlayerKey'].nunique()} "
        f"repetitions={args.repetitions} validation_used=false"
    )


if __name__ == "__main__":
    main()
