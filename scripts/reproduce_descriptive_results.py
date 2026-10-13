import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def save_bar(frame, x, y, hue, title, ylabel, path):
    categories = frame[x].drop_duplicates().tolist()
    hue_values = frame[hue].drop_duplicates().tolist()
    positions = range(len(categories))
    width = 0.36
    fig, axis = plt.subplots(figsize=(8, 5))
    for index, hue_value in enumerate(hue_values):
        values = (
            frame[frame[hue].eq(hue_value)]
            .set_index(x)[y]
            .reindex(categories, fill_value=0)
        )
        offsets = [position + (index - (len(hue_values) - 1) / 2) * width for position in positions]
        axis.bar(
            offsets,
            values,
            width=width,
            label=hue_value,
            color="white" if index == 0 else "black",
            edgecolor="black",
            hatch="" if index == 0 else "//",
        )
    axis.set_xticks(list(positions), categories)
    axis.set_title(title)
    axis.set_ylabel(ylabel)
    axis.legend(frameon=False)
    for spine in axis.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    injuries = pd.read_csv(args.raw_dir / "InjuryRecord.csv")
    plays = pd.read_csv(args.raw_dir / "PlayList.csv")

    exposure = (
        plays[["PlayerKey", "GameID", "FieldType"]]
        .drop_duplicates(["PlayerKey", "GameID"])
        .groupby("FieldType")
        .size()
        .rename("player_games")
    )
    injury_counts = injuries.groupby("Surface").size().rename("injuries")
    rates = pd.concat([exposure, injury_counts], axis=1).fillna(0).reset_index(names="surface")
    rates["injuries_per_1000_player_games"] = rates["injuries"] / rates["player_games"] * 1000
    rates.to_csv(args.output_dir / "injury_rates_by_surface.csv", index=False)

    severity_columns = ["DM_M1", "DM_M7", "DM_M28", "DM_M42"]
    severity = (
        injuries.groupby("Surface")[severity_columns]
        .sum()
        .reset_index()
        .melt(id_vars="Surface", var_name="threshold", value_name="injuries")
    )
    severity.to_csv(args.output_dir / "severity_by_surface.csv", index=False)

    body_part = (
        injuries.groupby(["BodyPart", "Surface"])
        .size()
        .rename("injuries")
        .reset_index()
    )
    body_part.to_csv(args.output_dir / "body_part_by_surface.csv", index=False)

    rate_plot = rates.rename(columns={"surface": "Surface"})
    rate_plot["measure"] = "Rate"
    save_bar(
        rate_plot,
        "measure",
        "injuries_per_1000_player_games",
        "Surface",
        "Injury rate by playing surface",
        "Injuries per 1,000 player-games",
        args.output_dir / "injury_rate_by_surface.png",
    )
    save_bar(
        severity,
        "threshold",
        "injuries",
        "Surface",
        "Cumulative injury severity by surface",
        "Injury records",
        args.output_dir / "severity_by_surface.png",
    )
    save_bar(
        body_part,
        "BodyPart",
        "injuries",
        "Surface",
        "Injury location by surface",
        "Injury records",
        args.output_dir / "body_part_by_surface.png",
    )

    summary = {
        "injury_rows": int(len(injuries)),
        "player_games": int(exposure.sum()),
        "severity_totals": {
            column: int(injuries[column].sum()) for column in severity_columns
        },
        "rate_denominator": "unique player-games",
    }
    (args.output_dir / "descriptive_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
