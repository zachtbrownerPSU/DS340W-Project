import argparse
import json
from pathlib import Path

import pandas as pd


EXPECTED_COLUMNS = {
    "InjuryRecord.csv": [
        "PlayerKey",
        "GameID",
        "PlayKey",
        "BodyPart",
        "Surface",
        "DM_M1",
        "DM_M7",
        "DM_M28",
        "DM_M42",
    ],
    "PlayList.csv": [
        "PlayerKey",
        "GameID",
        "PlayKey",
        "RosterPosition",
        "PlayerDay",
        "PlayerGame",
        "StadiumType",
        "FieldType",
        "Temperature",
        "Weather",
        "PlayType",
        "PlayerGamePlay",
        "Position",
        "PositionGroup",
    ],
    "PlayerTrackData.csv": [
        "PlayKey",
        "time",
        "event",
        "x",
        "y",
        "dir",
        "dis",
        "o",
        "s",
    ],
}


def check_columns(path: Path, expected: list[str]):
    columns = pd.read_csv(path, nrows=0).columns.tolist()
    if columns != expected:
        raise ValueError(f"Unexpected columns in {path.name}: {columns}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    for name, columns in EXPECTED_COLUMNS.items():
        path = args.raw_dir / name
        if not path.exists():
            raise FileNotFoundError(f"Missing required file: {path}")
        check_columns(path, columns)

    injuries = pd.read_csv(args.raw_dir / "InjuryRecord.csv")
    plays = pd.read_csv(args.raw_dir / "PlayList.csv")

    severity_columns = ["DM_M1", "DM_M7", "DM_M28", "DM_M42"]
    severity = {column: int(injuries[column].sum()) for column in severity_columns}
    parent_figure_2 = {
        "DM_M1": 437,
        "DM_M7": 339,
        "DM_M28": 170,
        "DM_M42": 131,
    }

    injury_games = injuries[["PlayerKey", "GameID"]].drop_duplicates()
    play_games = plays[["PlayerKey", "GameID"]].drop_duplicates()
    unmatched_games = injury_games.merge(
        play_games,
        on=["PlayerKey", "GameID"],
        how="left",
        indicator=True,
    )
    unmatched_games = int((unmatched_games["_merge"] != "both").sum())

    report = {
        "files": {
            name: {
                "bytes": (args.raw_dir / name).stat().st_size,
                "columns": columns,
            }
            for name, columns in EXPECTED_COLUMNS.items()
        },
        "players": int(plays["PlayerKey"].nunique()),
        "injured_players": int(injuries["PlayerKey"].nunique()),
        "player_games": int(play_games.shape[0]),
        "injury_rows": int(injuries.shape[0]),
        "injury_player_games": int(injury_games.shape[0]),
        "injuries_missing_play_key": int(injuries["PlayKey"].isna().sum()),
        "injury_games_missing_from_playlist": unmatched_games,
        "player_plays": int(plays.shape[0]),
        "duplicate_playlist_play_keys": int(plays["PlayKey"].duplicated().sum()),
        "severity_counts": severity,
        "parent_figure_2_counts": parent_figure_2,
        "parent_figure_2_difference": {
            key: parent_figure_2[key] - severity[key] for key in severity
        },
        "injuries_by_surface": {
            str(key): int(value)
            for key, value in injuries["Surface"].value_counts().items()
        },
        "injuries_by_body_part": {
            str(key): int(value)
            for key, value in injuries["BodyPart"].value_counts().items()
        },
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "data_audit.json"
    json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# Data audit",
        "",
        f"Players: {report['players']}",
        f"Injured players: {report['injured_players']}",
        f"Player-games: {report['player_games']}",
        f"Injury rows: {report['injury_rows']}",
        f"Injury player-games: {report['injury_player_games']}",
        f"Injuries without a PlayKey: {report['injuries_missing_play_key']}",
        "",
        "## Severity counts",
        "",
        "| Threshold | Official file | Parent Figure 2 | Difference |",
        "| --- | ---: | ---: | ---: |",
    ]
    for key in severity_columns:
        lines.append(
            f"| {key} | {severity[key]} | {parent_figure_2[key]} | "
            f"{parent_figure_2[key] - severity[key]} |"
        )
    lines.extend(
        [
            "",
            "The paper does not identify a direct data link. Its population description matches the NFL 1st and Future Analytics dataset, but the Figure 2 counts cannot be reproduced from InjuryRecord.csv without a different counting rule or additional data.",
            "",
        ]
    )
    (args.output_dir / "data_audit.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
