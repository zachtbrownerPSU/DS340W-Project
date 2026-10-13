import argparse
import csv
import json
import random
from pathlib import Path


def read_player_keys(path: Path):
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return {row["PlayerKey"] for row in csv.DictReader(handle)}


def partition(keys: set[str], seed: int):
    values = sorted(keys)
    random.Random(seed).shuffle(values)
    train_end = round(len(values) * 0.70)
    test_end = train_end + round(len(values) * 0.20)
    return values[:train_end], values[train_end:test_end], values[test_end:]


def write_split(path: Path, keys: list[str], injured: set[str]):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["PlayerKey", "ever_injured"])
        writer.writerows((key, int(key in injured)) for key in sorted(keys))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=340)
    args = parser.parse_args()

    players = read_player_keys(args.raw_dir / "PlayList.csv")
    injured = read_player_keys(args.raw_dir / "InjuryRecord.csv")
    if not injured <= players:
        raise ValueError("InjuryRecord.csv contains a PlayerKey absent from PlayList.csv")

    injured_train, injured_test, injured_validation = partition(injured, args.seed)
    healthy_train, healthy_test, healthy_validation = partition(
        players - injured, args.seed + 1
    )

    splits = {
        "train": injured_train + healthy_train,
        "test": injured_test + healthy_test,
        "validation": injured_validation + healthy_validation,
    }

    split_sets = {name: set(keys) for name, keys in splits.items()}
    if set.union(*split_sets.values()) != players:
        raise RuntimeError("Split files do not cover every player")
    if any(
        split_sets[left] & split_sets[right]
        for left, right in (("train", "test"), ("train", "validation"), ("test", "validation"))
    ):
        raise RuntimeError("A player appears in more than one split")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, keys in splits.items():
        write_split(args.output_dir / f"{name}_player_keys.csv", keys, injured)

    with (args.output_dir / "player_split_assignments.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(["PlayerKey", "split", "ever_injured"])
        for name, keys in splits.items():
            writer.writerows((key, name, int(key in injured)) for key in sorted(keys))

    summary = {
        "seed": args.seed,
        "grouping_unit": "PlayerKey",
        "total_players": len(players),
        "splits": {
            name: {
                "players": len(keys),
                "injured_players": sum(key in injured for key in keys),
                "noninjured_players": sum(key not in injured for key in keys),
            }
            for name, keys in splits.items()
        },
    }
    (args.output_dir / "split_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
