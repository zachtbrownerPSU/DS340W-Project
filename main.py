import argparse
import importlib.util
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def run(script, *arguments):
    command = [sys.executable, str(ROOT / "scripts" / script), *map(str, arguments)]
    print("Running " + script, flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def main():
    parser = argparse.ArgumentParser(description="Run the NFL injury prediction study.")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", choices=["cpu", "cuda:0", "cuda:1"], default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--chunksize", type=int, default=250000)
    parser.add_argument("--strict-paper-match", action="store_true")
    args = parser.parse_args()
    if args.threads < 1 or args.chunksize < 1:
        parser.error("Threads and chunk size must be positive")
    args.raw_dir = (ROOT / args.raw_dir).resolve()
    if args.output_dir is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
        args.output_dir = ROOT / "outputs" / ("run_" + stamp)
    else:
        args.output_dir = (ROOT / args.output_dir).resolve()
    if args.output_dir.exists():
        parser.error("Output directory already exists; choose a new directory")
    for filename in ["InjuryRecord.csv", "PlayList.csv", "PlayerTrackData.csv"]:
        if not (args.raw_dir / filename).is_file():
            parser.error("Missing data file: " + str(args.raw_dir / filename))
    split_file = ROOT / "splits/player_split_assignments.csv"
    if not split_file.is_file():
        parser.error("Missing committed player split assignments")
    packages = ["numpy", "pandas", "scipy", "sklearn", "xgboost", "matplotlib", "joblib"]
    missing = [name for name in packages if importlib.util.find_spec(name) is None]
    if missing:
        parser.error("Missing packages: " + ", ".join(missing) + ". Install requirements.txt first")
    environment = subprocess.run(
        [sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True, check=True
    )
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "packages.txt").write_text(environment.stdout)
    (args.output_dir / "started.json").write_text(json.dumps({
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "arguments": {k: str(v) for k, v in vars(args).items()},
        "scope": "Full feature-group study with original fixed splits and model settings",
    }, indent=2) + "\n")
    print("Results directory: " + str(args.output_dir), flush=True)
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
    for name in ["primary", "sensitivity"]:
        run("plot_study_results.py", "--run-dir", args.output_dir / name, "--output-dir", args.output_dir / name / "figures")
    comparison = args.output_dir / "paper_comparison.json"
    options = [] if args.strict_paper_match else ["--report-only"]
    run("check_paper_results.py", "--primary", args.output_dir / "primary", "--sensitivity", args.output_dir / "sensitivity", "--output", comparison, *options)
    matched = json.loads(comparison.read_text())["passed"]
    print((args.output_dir / "SUMMARY.md").read_text(), flush=True)
    print("Study completed. Results: " + str(args.output_dir), flush=True)
    if matched:
        print("Paper comparison passed.", flush=True)
    else:
        print("Paper comparison did not pass. Read paper_comparison.json before using these results as a match to the paper.", flush=True)


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as error:
        print("Pipeline stopped because a stage failed. Review the error above.", file=sys.stderr)
        sys.exit(error.returncode or 1)
