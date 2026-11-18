import contextlib
import importlib.util
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ReproductionTests(unittest.TestCase):
    def test_missing_data_stops_before_creating_output(self):
        module = load("reproduce_project")
        previous = Path.cwd()
        try:
            os.chdir(ROOT)
            with tempfile.TemporaryDirectory() as directory:
                output = Path(directory) / "run"
                argv = ["reproduce_project", "--raw-dir", str(Path(directory) / "missing"), "--output-dir", str(output)]
                with patch.object(sys, "argv", argv), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    module.main()
                self.assertFalse(output.exists())
        finally:
            os.chdir(previous)

    def test_runner_keeps_the_full_study_settings(self):
        module = load("reproduce_project")
        previous = Path.cwd()
        try:
            os.chdir(ROOT)
            with tempfile.TemporaryDirectory() as directory:
                raw = Path(directory) / "raw"
                raw.mkdir()
                for name in ["InjuryRecord.csv", "PlayList.csv", "PlayerTrackData.csv"]:
                    (raw / name).touch()
                output = Path(directory) / "run"
                argv = ["reproduce_project", "--raw-dir", str(raw), "--output-dir", str(output)]
                completed = subprocess.CompletedProcess([], 0, stdout="test-package==1\n", stderr="")
                with patch.object(sys, "argv", argv), patch.object(module.subprocess, "run", return_value=completed), patch.object(module, "run") as called, contextlib.redirect_stdout(io.StringIO()):
                    module.main()
                commands = [call.args for call in called.call_args_list]
                self.assertEqual(commands[0][0], "validate_data.py")
                self.assertEqual(commands[-1][0], "check_paper_results.py")
                fits = [c for c in commands if c[0] == "run_feature_ablation.py"]
                self.assertEqual(len(fits), 2)
                for command in fits:
                    for flag, expected in [("--repeats", 5), ("--folds", 5), ("--bootstrap", 2000), ("--seed", 340), ("--device", "cpu")]:
                        self.assertEqual(command[command.index(flag) + 1], expected)
                self.assertEqual(fits[0][-1], "exclude_players")
                self.assertEqual(fits[1][-1], "game_order")
                self.assertTrue((output / "packages.txt").is_file())
                with patch.object(sys, "argv", argv), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    module.main()
        finally:
            os.chdir(previous)

    def test_comparison_rejects_an_incorrect_metric(self):
        module = load("check_paper_results")
        source = json.loads((ROOT / "reference/paper_results.json").read_text())["primary"]
        metrics = []
        for record in source["performance"]:
            metrics.extend([record.copy() for _ in range(5)])
        pairs = [{**r, "metric": "average_precision"} for r in source["paired_average_precision"]]

        def read(path):
            return metrics if path.name == "repeat_metrics.csv" else pairs

        with patch.object(module, "rows", side_effect=read), patch.object(Path, "read_text", return_value=json.dumps(source["cohort"])):
            self.assertTrue(all(r["passed"] for r in module.compare(Path("run"), source, 0.00005)))
            metrics[0]["average_precision"] += 0.1
            self.assertTrue(any(not r["passed"] for r in module.compare(Path("run"), source, 0.00005)))


if __name__ == "__main__":
    unittest.main()
