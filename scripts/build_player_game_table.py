import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd


STATIC_COLUMNS = [
    "RosterPosition",
    "PlayerDay",
    "PlayerGame",
    "StadiumType",
    "FieldType",
    "Temperature",
    "Weather",
    "Position",
    "PositionGroup",
]

GAME_WORKLOAD_COLUMNS = [
    "game_plays",
    "game_tracked_plays",
    "game_distance",
    "game_mean_speed",
    "game_max_speed",
    "game_high_speed_distance",
    "game_direction_changes",
    "game_acceleration_max",
    "game_deceleration_min",
]


def first_value(series: pd.Series):
    values = series.dropna()
    return values.iloc[0] if not values.empty else np.nan


def clean_stadium(value):
    if pd.isna(value):
        return "Unknown"
    text = re.sub(r"[^a-z ]", " ", str(value).lower())
    text = re.sub(r"\s+", " ", text).strip()
    if any(word in text for word in ["indoor", "dome", "closed"]):
        return "Indoor"
    if "outdoor" in text or "open" in text or "outside" in text:
        return "Outdoor"
    if "retractable" in text:
        return "Retractable"
    return "Other"


def clean_weather(value):
    if pd.isna(value):
        return "Unknown"
    text = re.sub(r"[^a-z ]", " ", str(value).lower())
    text = re.sub(r"\s+", " ", text).strip()
    if any(word in text for word in ["indoor", "controlled", "climate"]):
        return "Indoor"
    if any(word in text for word in ["snow", "sleet", "flurr"]):
        return "Snow"
    if any(word in text for word in ["rain", "shower", "wet"]):
        return "Rain"
    if any(word in text for word in ["cloud", "overcast", "hazy", "fog"]):
        return "Cloudy"
    if any(word in text for word in ["clear", "sun", "fair"]):
        return "Clear"
    return "Other"


def aggregate_games(plays: pd.DataFrame):
    static = (
        plays.groupby(["PlayerKey", "GameID"], sort=False)[STATIC_COLUMNS]
        .agg(first_value)
        .reset_index()
    )
    workload = (
        plays.groupby(["PlayerKey", "GameID"], sort=False)
        .agg(
            game_plays=("PlayKey", "size"),
            game_tracked_plays=("tracking_frames", "count"),
            game_distance=("distance_total", "sum"),
            game_mean_speed=("speed_mean", "mean"),
            game_max_speed=("speed_max", "max"),
            game_high_speed_distance=("high_speed_distance", "sum"),
            game_direction_changes=("sharp_direction_changes", "sum"),
            game_acceleration_max=("acceleration_max", "max"),
            game_deceleration_min=("deceleration_min", "min"),
        )
        .reset_index()
    )
    return static.merge(workload, on=["PlayerKey", "GameID"], validate="one_to_one")


def add_history(games: pd.DataFrame):
    games = games.sort_values(["PlayerKey", "PlayerGame", "PlayerDay", "GameID"]).copy()
    grouped = games.groupby("PlayerKey", sort=False)
    games["games_before"] = grouped.cumcount()
    games["days_since_previous_game"] = grouped["PlayerDay"].diff()
    games["prior_injury_count"] = grouped["injury"].cumsum() - games["injury"]

    for column in GAME_WORKLOAD_COLUMNS:
        previous = grouped[column].shift(1)
        games[f"prior_{column[5:]}"] = previous
        games[f"prior_3_{column[5:]}"] = previous.groupby(games["PlayerKey"]).transform(
            lambda values: values.rolling(3, min_periods=1).mean()
        )
    return games


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--tracking-features", type=Path, required=True)
    parser.add_argument("--split-file", type=Path, required=True)
    parser.add_argument("--output-file", type=Path, required=True)
    args = parser.parse_args()

    plays = pd.read_csv(args.raw_dir / "PlayList.csv")
    injuries = pd.read_csv(args.raw_dir / "InjuryRecord.csv")
    tracking = pd.read_csv(args.tracking_features)
    splits = pd.read_csv(args.split_file)

    if tracking["PlayKey"].duplicated().any():
        raise ValueError("Tracking feature file contains duplicate PlayKey values")
    if splits["PlayerKey"].duplicated().any():
        raise ValueError("Split assignment file contains duplicate PlayerKey values")

    plays = plays.merge(tracking, on="PlayKey", how="left", validate="one_to_one")
    games = aggregate_games(plays)

    injury_games = (
        injuries.groupby(["PlayerKey", "GameID"], as_index=False)
        .agg(
            injury=("GameID", "size"),
            DM_M1=("DM_M1", "max"),
            DM_M7=("DM_M7", "max"),
            DM_M28=("DM_M28", "max"),
            DM_M42=("DM_M42", "max"),
        )
    )
    injury_games["injury"] = 1
    games = games.merge(injury_games, on=["PlayerKey", "GameID"], how="left")
    for column in ["injury", "DM_M1", "DM_M7", "DM_M28", "DM_M42"]:
        games[column] = games[column].fillna(0).astype("int8")

    games["Temperature"] = pd.to_numeric(games["Temperature"], errors="coerce")
    games.loc[games["Temperature"].eq(-999), "Temperature"] = np.nan
    games["StadiumType_clean"] = games["StadiumType"].map(clean_stadium)
    games["Weather_clean"] = games["Weather"].map(clean_weather)
    games = add_history(games)
    games = games.merge(
        splits[["PlayerKey", "split"]],
        on="PlayerKey",
        how="left",
        validate="many_to_one",
    )

    if games["split"].isna().any():
        raise RuntimeError("At least one player-game has no split assignment")
    if int(games["injury"].sum()) != int(injury_games["injury"].sum()):
        raise RuntimeError("Player-game table lost an injury game")

    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    games.to_csv(args.output_file, index=False)
    print(
        f"saved={args.output_file} player_games={len(games)} "
        f"injury_games={int(games['injury'].sum())} tracked_games={int((games['game_tracked_plays'] > 0).sum())}"
    )


if __name__ == "__main__":
    main()
