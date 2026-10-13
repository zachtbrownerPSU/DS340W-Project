import argparse
from pathlib import Path

import numpy as np
import pandas as pd


USE_COLUMNS = ["PlayKey", "time", "x", "y", "dir", "dis", "s"]


def aggregate_plays(frame: pd.DataFrame):
    frame = frame.sort_values(["PlayKey", "time"])
    groups = frame.groupby("PlayKey", sort=False)
    speed_change = groups["s"].diff()
    time_change = groups["time"].diff()
    acceleration = speed_change.div(time_change.where(time_change > 0))
    direction_change = groups["dir"].diff()
    direction_change = ((direction_change + 180) % 360 - 180).abs()

    frame = frame.assign(
        acceleration=acceleration,
        direction_change=direction_change,
        high_speed_distance=frame["dis"].where(frame["s"] >= 6, 0),
        high_speed_frame=(frame["s"] >= 6).astype("int8"),
        sharp_direction_change=(direction_change >= 45).astype("int8"),
    )
    groups = frame.groupby("PlayKey", sort=False)
    result = groups.agg(
        tracking_frames=("time", "size"),
        time_min=("time", "min"),
        time_max=("time", "max"),
        distance_total=("dis", "sum"),
        speed_mean=("s", "mean"),
        speed_max=("s", "max"),
        speed_std=("s", "std"),
        acceleration_mean=("acceleration", "mean"),
        acceleration_max=("acceleration", "max"),
        deceleration_min=("acceleration", "min"),
        direction_change_mean=("direction_change", "mean"),
        direction_change_max=("direction_change", "max"),
        direction_change_total=("direction_change", "sum"),
        sharp_direction_changes=("sharp_direction_change", "sum"),
        high_speed_distance=("high_speed_distance", "sum"),
        high_speed_frames=("high_speed_frame", "sum"),
        x_min=("x", "min"),
        x_max=("x", "max"),
        y_min=("y", "min"),
        y_max=("y", "max"),
    ).reset_index()
    result["duration_seconds"] = result["time_max"] - result["time_min"]
    result["x_range"] = result["x_max"] - result["x_min"]
    result["y_range"] = result["y_max"] - result["y_min"]
    return result.drop(columns=["time_min", "time_max", "x_min", "x_max", "y_min", "y_max"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--track-file", type=Path, required=True)
    parser.add_argument("--output-file", type=Path, required=True)
    parser.add_argument("--chunksize", type=int, default=1_000_000)
    args = parser.parse_args()

    results = []
    carry = pd.DataFrame()
    rows_read = 0

    reader = pd.read_csv(
        args.track_file,
        usecols=USE_COLUMNS,
        chunksize=args.chunksize,
        dtype={"PlayKey": "string"},
    )
    for number, chunk in enumerate(reader, start=1):
        rows_read += len(chunk)
        if not carry.empty:
            chunk = pd.concat([carry, chunk], ignore_index=True)

        last_key = chunk["PlayKey"].iloc[-1]
        carry_mask = chunk["PlayKey"].eq(last_key)
        carry = chunk.loc[carry_mask].copy()
        complete = chunk.loc[~carry_mask]
        if not complete.empty:
            results.append(aggregate_plays(complete))
        print(f"chunk={number} rows={rows_read} completed_plays={sum(len(x) for x in results)}")

    if not carry.empty:
        results.append(aggregate_plays(carry))

    features = pd.concat(results, ignore_index=True)
    if features["PlayKey"].duplicated().any():
        duplicates = int(features["PlayKey"].duplicated().sum())
        raise RuntimeError(f"Tracking aggregation produced {duplicates} duplicate PlayKey values")

    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(args.output_file, index=False)
    print(f"saved={args.output_file} plays={len(features)} rows={rows_read}")


if __name__ == "__main__":
    main()
