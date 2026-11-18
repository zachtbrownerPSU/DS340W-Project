import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def run(script, *arguments):
    command = [sys.executable, str(Path("scripts") / script), *map(str, arguments)]
    print("Running " + script, flush=True)
    subprocess.run(command, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=["cpu", "cuda:0", "cuda:1"], default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--chunksize", type=int, default=250000)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if Path.cwd().resolve() != root:
        parser.error("Run this command from the repository directory")
    if args.output_dir.exists():
        parser.error("Output directory already exists; choose a new directory")
    if args.threads < 1 or args.chunksize < 1:
        parser.error("Threads and chunk size must be positive")
    for filename in ["InjuryRecord.csv", "PlayList.csv", "PlayerTrackData.csv"]:
        if not (args.raw_dir / filename).is_file():
            parser.error("Missing data file: " + str(args.raw_dir / filename))
    split_file = Path("splits/player_split_assignments.csv")
    if not split_file.is_file():
        parser.error("Missing committed player split assignments")
    args.output_dir.mkdir(parents=True)
    environment = subprocess.run(
        [sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True, check=True
    )
    (args.output_dir / "packages.txt").write_text(environment.stdout)
    (args.output_dir / "started.json").write_text(json.dumps({
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "arguments": {k: str(v) for k, v in vars(args).items()},
        "scope": "Full feature-group study with original fixed splits and model settings",
    }, indent=2) + "\n")
    processed = args.output_dir / "processed"
    tracking = processed / "play_tracking_features.csv"
    table = processed / "player_game_modeling.csv"
    run("validate_data.py", "--raw-dir", args.raw_dir, "--output-dir", args.output_dir / "audit")
    run("build_tracking_features.py", "--track-file", args.raw_dir / "PlayerTrackData.csv", "--output-file", tracking, "--chunksize", args.chunksize)
    run("build_player_game_table.py", "--raw-dir", args.raw_dir, "--tracking-features", tracking, "--split-file", split_file, "--output-file", table)
    run("reproduce_descriptive_results.py", "--raw-dir", args.raw_dir, "--output-dir", args.output_dir / "descriptive")
    for policy, name in [("exclude_players", "primary"), ("game_order", "sensitivity")]:
        run("run_feature_ablation.py", "--table", table, "--output-dir", args.output_dir / name, "--device", args.device, "--threads", args.threads, "--repeats", 5, "--folds", 5, "--bootstrap", 2000, "--seed", 340, "--chronology-policy", policy)
    run("summarize_feature_ablation.py", "--primary", args.output_dir / "primary", "--sensitivity", args.output_dir / "sensitivity", "--output", args.output_dir / "SUMMARY.md")
    run("check_paper_results.py", "--primary", args.output_dir / "primary", "--sensitivity", args.output_dir / "sensitivity", "--output", args.output_dir / "paper_comparison.json")
    print("Study completed and paper comparison passed. Results: " + str(args.output_dir), flush=True)


if __name__ == "__main__":
    main()
