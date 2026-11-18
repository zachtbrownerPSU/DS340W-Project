import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean


def rows(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def compare(root, expected, tolerance):
    audit = json.loads((root / "data_audit.json").read_text())
    checks = []
    for key, value in expected["cohort"].items():
        checks.append({"item": key, "expected": value, "actual": audit.get(key), "passed": audit.get(key) == value})
    grouped = defaultdict(list)
    for row in rows(root / "repeat_metrics.csv"):
        grouped[(row["model"], row["feature_set"])].append(row)
    for record in expected["performance"]:
        values = grouped.get((record["model"], record["feature_set"]), [])
        for metric in ["average_precision", "roc_auc", "brier_score"]:
            actual = mean(float(row[metric]) for row in values) if values else None
            checks.append({"item": record["model"] + "/" + record["feature_set"] + "/" + metric, "expected": record[metric], "actual": actual, "passed": len(values) == 5 and actual is not None and abs(actual - record[metric]) <= tolerance})
    observed = {(r["model"], r["comparison"], r["metric"]): r for r in rows(root / "player_bootstrap_differences.csv")}
    for record in expected["paired_average_precision"]:
        actual = observed.get((record["model"], record["comparison"], "average_precision"))
        for metric in ["observed_difference", "lower_95", "upper_95"]:
            value = float(actual[metric]) if actual else None
            checks.append({"item": record["model"] + "/" + record["comparison"] + "/" + metric, "expected": record[metric], "actual": value, "passed": value is not None and abs(value - record[metric]) <= tolerance})
    return checks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--primary", type=Path, required=True)
    parser.add_argument("--sensitivity", type=Path, required=True)
    parser.add_argument("--expected", type=Path, default=Path(__file__).resolve().parents[1] / "reference/paper_results.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    expected = json.loads(args.expected.read_text())
    tolerance = expected["absolute_tolerance"]
    checked = {name: compare(root, expected[name], tolerance) for name, root in [("primary", args.primary), ("sensitivity", args.sensitivity)]}
    failures = [(name, row) for name, records in checked.items() for row in records if not row["passed"]]
    report = {"passed": not failures, "absolute_tolerance": tolerance, "scope": "Reported feature-group metrics and paired AP intervals; run the separate prediction verifier as well", "checks": checked}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print("Paper comparison: " + ("PASS" if not failures else "FAIL"))
    for name, row in failures:
        print(name + " " + row["item"] + ": expected " + str(row["expected"]) + ", found " + str(row["actual"]))
    if failures:
        raise SystemExit("Results do not match the recorded study within tolerance; inspect data, package versions, folds, and device before recording")


if __name__ == "__main__":
    main()
