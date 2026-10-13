import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from build_player_game_table import clean_stadium, clean_weather


TRACKING_COLUMNS = [
    "distance_total",
    "speed_mean",
    "speed_max",
    "speed_std",
    "acceleration_max",
    "deceleration_min",
    "direction_change_total",
    "sharp_direction_changes",
    "high_speed_distance",
    "duration_seconds",
]

ROLLING_COLUMNS = [
    "distance_total",
    "speed_mean",
    "speed_max",
    "acceleration_max",
    "deceleration_min",
    "sharp_direction_changes",
    "high_speed_distance",
]


def add_sequence_features(frame):
    frame = frame.sort_values(
        ["PlayerKey", "PlayerGame", "GameID", "PlayerGamePlay"]
    ).copy()
    groups = frame.groupby(["PlayerKey", "GameID"], sort=False)
    frame["plays_completed"] = groups.cumcount() + 1

    for column in ROLLING_COLUMNS:
        frame[f"recent_5_{column}"] = groups[column].transform(
            lambda values: values.rolling(5, min_periods=1).mean()
        )

    for column in [
        "distance_total",
        "sharp_direction_changes",
        "high_speed_distance",
    ]:
        frame[f"game_to_date_{column}"] = groups[column].cumsum()

    return frame


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--tracking-features", type=Path, required=True)
    parser.add_argument("--split-file", type=Path, required=True)
    parser.add_argument("--output-file", type=Path, required=True)
    args = parser.parse_args()

    plays = pd.read_csv(args.raw_dir / "PlayList.csv")
    tracking = pd.read_csv(args.tracking_features)
    injuries = pd.read_csv(args.raw_dir / "InjuryRecord.csv")
    splits = pd.read_csv(args.split_file)

    plays = plays.merge(tracking, on="PlayKey", how="left", validate="one_to_one")
    injury_timing = injuries.dropna(subset=["PlayKey"])[
        ["PlayerKey", "GameID", "PlayKey"]
    ].drop_duplicates().merge(
        plays[["PlayerKey", "GameID", "PlayKey", "PlayerGamePlay"]],
        on=["PlayerKey", "GameID", "PlayKey"],
        how="left",
        validate="one_to_one",
    )
    if injury_timing["PlayerGamePlay"].isna().any():
        missing = int(injury_timing["PlayerGamePlay"].isna().sum())
        raise RuntimeError(f"Could not locate {missing} recorded injury plays")

    injury_play_number = (
        injury_timing.groupby(["PlayerKey", "GameID"])["PlayerGamePlay"]
        .min()
        .rename("injury_play_number")
        .reset_index()
    )
    unknown_timing = injuries[injuries["PlayKey"].isna()][
        ["PlayerKey", "GameID"]
    ].drop_duplicates()
    unknown_timing["unknown_injury_timing"] = 1

    plays = plays.merge(
        injury_play_number,
        on=["PlayerKey", "GameID"],
        how="left",
        validate="many_to_one",
    )
    plays = plays.merge(
        unknown_timing,
        on=["PlayerKey", "GameID"],
        how="left",
        validate="many_to_one",
    )
    plays = plays[plays["unknown_injury_timing"].isna()].copy()

    distance_to_injury = plays["injury_play_number"] - plays["PlayerGamePlay"]
    plays["plays_until_injury"] = distance_to_injury
    for window in [5, 10, 20]:
        plays[f"injury_within_{window}_plays"] = (
            distance_to_injury.between(1, window).astype("int8")
        )

    plays = plays[
        plays["injury_play_number"].isna()
        | plays["PlayerGamePlay"].lt(plays["injury_play_number"])
    ].copy()
    plays["Temperature"] = pd.to_numeric(plays["Temperature"], errors="coerce")
    plays.loc[plays["Temperature"].eq(-999), "Temperature"] = np.nan
    plays["StadiumType_clean"] = plays["StadiumType"].map(clean_stadium)
    plays["Weather_clean"] = plays["Weather"].map(clean_weather)

    for column in TRACKING_COLUMNS:
        plays[column] = pd.to_numeric(plays[column], errors="coerce")
    plays = add_sequence_features(plays)
    plays = plays.merge(
        splits[["PlayerKey", "split"]],
        on="PlayerKey",
        how="left",
        validate="many_to_one",
    )
    if plays["split"].isna().any():
        raise RuntimeError("At least one play has no split assignment")

    keep = [
        "PlayerKey",
        "GameID",
        "PlayKey",
        "split",
        "RosterPosition",
        "PlayerDay",
        "PlayerGame",
        "FieldType",
        "Temperature",
        "PlayType",
        "PlayerGamePlay",
        "Position",
        "PositionGroup",
        "StadiumType_clean",
        "Weather_clean",
        "plays_completed",
        "injury_play_number",
        "plays_until_injury",
        "injury_within_5_plays",
        "injury_within_10_plays",
        "injury_within_20_plays",
    ]
    keep.extend(TRACKING_COLUMNS)
    keep.extend(f"recent_5_{column}" for column in ROLLING_COLUMNS)
    keep.extend(
        [
            "game_to_date_distance_total",
            "game_to_date_sharp_direction_changes",
            "game_to_date_high_speed_distance",
        ]
    )
    plays = plays[keep]

    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    plays.to_csv(args.output_file, index=False)
    positives = {
        window: int(plays[f"injury_within_{window}_plays"].sum())
        for window in [5, 10, 20]
    }
    print(
        f"saved={args.output_file} rows={len(plays)} "
        f"players={plays['PlayerKey'].nunique()} positives={positives}"
    )


if __name__ == "__main__":
    main()
