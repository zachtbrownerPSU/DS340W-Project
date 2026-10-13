import argparse
import json
import re
from pathlib import Path

import pandas as pd


LOWER_BODY_PATTERN = re.compile(
    r"\b(?:knee|ankle|hamstring|groin|calf|calves|foot|feet|hip|hips|quadricep|quadriceps|quad|toe|toes|achilles|heel|shin|thigh|tibia|fibula|lower leg|adductor|glute|pelvis)\b",
    re.IGNORECASE,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", default="data/external/raw/nflverse")
    parser.add_argument("--output-dir", default="data/external/processed")
    parser.add_argument("--start-season", type=int, default=2020)
    parser.add_argument("--end-season", type=int, default=2024)
    return parser.parse_args()


def read_seasons(raw_dir, prefix, seasons):
    frames = []
    for season in seasons:
        path = raw_dir / f"{prefix}_{season}.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing required file: {path}")
        frames.append(pd.read_csv(path, low_memory=False))
    return pd.concat(frames, ignore_index=True)


def clean_id(series):
    return series.astype("string").str.strip().replace({"": pd.NA, "nan": pd.NA})


def clean_name(series):
    return (
        series.astype("string")
        .str.lower()
        .str.normalize("NFKD")
        .str.encode("ascii", errors="ignore")
        .str.decode("ascii")
        .str.replace(r"[^a-z0-9]", "", regex=True)
    )


def classify_surface(series):
    cleaned = series.astype("string").str.strip().str.lower()
    natural = cleaned.str.contains("grass", na=False)
    synthetic = cleaned.str.contains("turf", na=False)
    result = pd.Series("unknown", index=series.index, dtype="string")
    result.loc[natural] = "natural"
    result.loc[synthetic] = "synthetic"
    return result


def prepare_rosters(rosters):
    rosters = rosters.loc[rosters["game_type"].eq("REG")].copy()
    rosters["gsis_id"] = clean_id(rosters["gsis_id"])
    rosters["pfr_id"] = clean_id(rosters["pfr_id"])
    rosters["clean_name"] = clean_name(rosters["full_name"])
    exact_columns = [
        "season",
        "week",
        "team",
        "pfr_id",
        "gsis_id",
        "position",
        "height",
        "weight",
        "years_exp",
        "birth_date",
    ]
    exact = rosters[exact_columns].dropna(subset=["pfr_id", "gsis_id"])
    exact = exact.drop_duplicates(["season", "week", "team", "pfr_id"])
    name_map = rosters.dropna(subset=["gsis_id", "clean_name"])[
        ["season", "team", "clean_name", "gsis_id"]
    ].drop_duplicates()
    counts = name_map.groupby(["season", "team", "clean_name"])["gsis_id"].transform("nunique")
    name_map = name_map.loc[counts.eq(1)].drop_duplicates(["season", "team", "clean_name"])
    return exact, name_map


def map_players(snaps, exact, name_map):
    snaps = snaps.loc[snaps["game_type"].eq("REG")].copy()
    snaps["pfr_player_id"] = clean_id(snaps["pfr_player_id"])
    snaps["clean_name"] = clean_name(snaps["player"])
    merged = snaps.merge(
        exact,
        left_on=["season", "week", "team", "pfr_player_id"],
        right_on=["season", "week", "team", "pfr_id"],
        how="left",
        validate="many_to_one",
        suffixes=("", "_roster"),
    )
    merged["mapping_method"] = merged["gsis_id"].notna().map({True: "pfr_id", False: "unmapped"})
    missing = merged["gsis_id"].isna()
    fallback = merged.loc[missing, ["season", "team", "clean_name"]].merge(
        name_map,
        on=["season", "team", "clean_name"],
        how="left",
        validate="many_to_one",
    )
    merged.loc[missing, "gsis_id"] = fallback["gsis_id"].to_numpy()
    name_mapped = missing & merged["gsis_id"].notna()
    merged.loc[name_mapped, "mapping_method"] = "name"
    return merged


def build_team_schedule(games, seasons):
    games = games.loc[
        games["season"].isin(seasons) & games["game_type"].eq("REG")
    ].copy()
    shared = [
        "game_id",
        "season",
        "week",
        "gameday",
        "surface",
        "roof",
        "temp",
        "wind",
        "stadium_id",
        "stadium",
    ]
    home = games[shared + ["home_team", "away_team", "home_rest"]].rename(
        columns={"home_team": "team", "away_team": "opponent", "home_rest": "rest_days"}
    )
    home["is_home"] = 1
    away = games[shared + ["away_team", "home_team", "away_rest"]].rename(
        columns={"away_team": "team", "home_team": "opponent", "away_rest": "rest_days"}
    )
    away["is_home"] = 0
    schedule = pd.concat([home, away], ignore_index=True)
    schedule["gameday"] = pd.to_datetime(schedule["gameday"], errors="coerce")
    schedule["surface"] = schedule["surface"].astype("string").str.strip().str.lower()
    schedule["surface_group"] = classify_surface(schedule["surface"])
    schedule = schedule.sort_values(["season", "team", "gameday", "week"])
    schedule["next_team_week"] = schedule.groupby(["season", "team"])["week"].shift(-1)
    schedule["next_game_date"] = schedule.groupby(["season", "team"])["gameday"].shift(-1)
    return schedule


def prepare_injuries(injuries):
    injuries = injuries.loc[injuries["game_type"].eq("REG")].copy()
    injuries["gsis_id"] = clean_id(injuries["gsis_id"])
    text_columns = [
        "report_primary_injury",
        "report_secondary_injury",
        "practice_primary_injury",
        "practice_secondary_injury",
    ]
    combined = injuries[text_columns].fillna("").astype(str).agg(" | ".join, axis=1)
    injuries["lower_body_report"] = combined.str.contains(LOWER_BODY_PATTERN, na=False).astype(int)
    injuries["lower_body_out"] = (
        injuries["lower_body_report"].eq(1)
        & injuries["report_status"].fillna("").str.casefold().eq("out")
    ).astype(int)
    injuries["any_injury_report"] = injuries["gsis_id"].notna().astype(int)
    return injuries.groupby(["season", "team", "week", "gsis_id"], as_index=False).agg(
        lower_body_report=("lower_body_report", "max"),
        lower_body_out=("lower_body_out", "max"),
        any_injury_report=("any_injury_report", "max"),
    )


def add_workload_features(cohort):
    snap_columns = ["offense_snaps", "defense_snaps", "st_snaps"]
    for column in snap_columns:
        cohort[column] = pd.to_numeric(cohort[column], errors="coerce").fillna(0)
    cohort["total_snaps"] = cohort[snap_columns].sum(axis=1)
    cohort = cohort.sort_values(["season", "gsis_id", "gameday", "game_id"])
    groups = cohort.groupby(["season", "gsis_id"], sort=False)
    cohort["games_played_to_date"] = groups.cumcount() + 1
    cohort["season_snaps_to_date"] = groups["total_snaps"].cumsum()
    cohort["snaps_last_3_games"] = groups["total_snaps"].transform(
        lambda values: values.rolling(3, min_periods=1).sum()
    )
    cohort["snaps_last_5_games"] = groups["total_snaps"].transform(
        lambda values: values.rolling(5, min_periods=1).sum()
    )
    return cohort


def main():
    args = parse_args()
    raw_dir = Path(args.raw_dir)
    output_dir = Path(args.output_dir)
    seasons = list(range(args.start_season, args.end_season + 1))
    output_dir.mkdir(parents=True, exist_ok=True)

    injuries = read_seasons(raw_dir, "injuries", seasons)
    rosters = read_seasons(raw_dir, "roster_weekly", seasons)
    snaps = read_seasons(raw_dir, "snap_counts", seasons)
    games_path = raw_dir / "games.csv"
    if not games_path.exists():
        raise FileNotFoundError(f"Missing required file: {games_path}")
    games = pd.read_csv(games_path, low_memory=False)

    exact, name_map = prepare_rosters(rosters)
    cohort = map_players(snaps, exact, name_map)
    schedule = build_team_schedule(games, seasons)
    injury_weeks = prepare_injuries(injuries)

    cohort = cohort.merge(
        schedule,
        on=["game_id", "season", "week", "team", "opponent"],
        how="left",
        validate="many_to_one",
    )
    cohort = add_workload_features(cohort)

    current = injury_weeks.rename(
        columns={
            "lower_body_report": "current_lower_body_report",
            "lower_body_out": "current_lower_body_out",
            "any_injury_report": "current_any_injury_report",
        }
    )
    cohort = cohort.merge(
        current,
        on=["season", "team", "week", "gsis_id"],
        how="left",
        validate="many_to_one",
    )
    target = injury_weeks.rename(
        columns={
            "week": "next_team_week",
            "lower_body_report": "next_lower_body_report",
            "lower_body_out": "next_lower_body_out",
            "any_injury_report": "next_any_injury_report",
        }
    )
    cohort = cohort.merge(
        target,
        on=["season", "team", "next_team_week", "gsis_id"],
        how="left",
        validate="many_to_one",
    )
    report_columns = [
        "current_lower_body_report",
        "current_lower_body_out",
        "current_any_injury_report",
        "next_lower_body_report",
        "next_lower_body_out",
        "next_any_injury_report",
    ]
    cohort[report_columns] = cohort[report_columns].fillna(0).astype(int)
    cohort["new_lower_body_report"] = (
        cohort["next_lower_body_report"].eq(1)
        & cohort["current_lower_body_report"].eq(0)
    ).astype(int)
    cohort["new_lower_body_out"] = (
        cohort["next_lower_body_out"].eq(1)
        & cohort["current_lower_body_report"].eq(0)
    ).astype(int)

    mapped_before_filter = int(cohort["gsis_id"].notna().sum())
    total_before_filter = len(cohort)
    cohort = cohort.loc[
        cohort["gsis_id"].notna()
        & cohort["next_team_week"].notna()
        & cohort["current_lower_body_report"].eq(0)
    ].copy()
    cohort["next_team_week"] = cohort["next_team_week"].astype(int)

    keep = [
        "season",
        "game_id",
        "week",
        "gameday",
        "next_team_week",
        "next_game_date",
        "team",
        "opponent",
        "gsis_id",
        "pfr_player_id",
        "player",
        "position",
        "position_roster",
        "mapping_method",
        "height",
        "weight",
        "years_exp",
        "birth_date",
        "offense_snaps",
        "defense_snaps",
        "st_snaps",
        "total_snaps",
        "games_played_to_date",
        "season_snaps_to_date",
        "snaps_last_3_games",
        "snaps_last_5_games",
        "surface",
        "surface_group",
        "roof",
        "temp",
        "wind",
        "stadium_id",
        "stadium",
        "rest_days",
        "is_home",
        "current_any_injury_report",
        "next_any_injury_report",
        "next_lower_body_report",
        "next_lower_body_out",
        "new_lower_body_report",
        "new_lower_body_out",
    ]
    cohort = cohort[keep].sort_values(["season", "week", "game_id", "team", "gsis_id"])
    output_path = output_dir / "external_player_game_injury_cohort.csv"
    cohort.to_csv(output_path, index=False)

    game_cohort = cohort.groupby(
        [
            "season",
            "game_id",
            "week",
            "gameday",
            "surface",
            "surface_group",
            "roof",
            "stadium_id",
            "stadium",
        ],
        as_index=False,
        dropna=False,
    ).agg(
        at_risk_player_games=("gsis_id", "size"),
        new_lower_body_reports=("new_lower_body_report", "sum"),
        new_lower_body_out=("new_lower_body_out", "sum"),
        mean_total_snaps=("total_snaps", "mean"),
        mean_last_3_game_snaps=("snaps_last_3_games", "mean"),
    )
    game_output_path = output_dir / "external_game_surface_cohort.csv"
    game_cohort.to_csv(game_output_path, index=False)

    rates = cohort.groupby(["season", "surface_group"], as_index=False).agg(
        player_games=("new_lower_body_report", "size"),
        new_lower_body_reports=("new_lower_body_report", "sum"),
        new_lower_body_out=("new_lower_body_out", "sum"),
    )
    rates["reports_per_1000_player_games"] = (
        rates["new_lower_body_reports"] / rates["player_games"] * 1000
    )
    rates["out_reports_per_1000_player_games"] = (
        rates["new_lower_body_out"] / rates["player_games"] * 1000
    )
    rates_path = output_dir / "external_cohort_rates.csv"
    rates.to_csv(rates_path, index=False)

    summary = {
        "seasons": seasons,
        "raw_snap_rows": int(total_before_filter),
        "mapped_snap_rows": mapped_before_filter,
        "mapping_rate": mapped_before_filter / total_before_filter if total_before_filter else 0,
        "cohort_rows": int(len(cohort)),
        "players": int(cohort["gsis_id"].nunique()),
        "games": int(cohort["game_id"].nunique()),
        "new_lower_body_reports": int(cohort["new_lower_body_report"].sum()),
        "new_lower_body_out_reports": int(cohort["new_lower_body_out"].sum()),
        "outcome_rate": float(cohort["new_lower_body_report"].mean()),
        "surface_counts": cohort["surface_group"].value_counts(dropna=False).to_dict(),
        "output": str(output_path),
        "game_output": str(game_output_path),
        "rates_output": str(rates_path),
    }
    summary_path = output_dir / "external_cohort_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
