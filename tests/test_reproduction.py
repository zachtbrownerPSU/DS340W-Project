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
    path = ROOT / "main.py" if name == "main" else ROOT / "scripts" / (name + ".py")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ReproductionTests(unittest.TestCase):
    def test_missing_data_stops_before_creating_output(self):
        module = load("main")
        previous = Path.cwd()
        try:
            os.chdir(ROOT)
            with tempfile.TemporaryDirectory() as directory:
                output = Path(directory) / "run"
                argv = ["main.py", "--raw-dir", str(Path(directory) / "missing"), "--output-dir", str(output)]
                with patch.object(sys, "argv", argv), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    module.main()
                self.assertFalse(output.exists())
        finally:
            os.chdir(previous)

    def test_runner_keeps_the_full_study_settings(self):
        module = load("main")
        previous = Path.cwd()
        try:
            os.chdir(ROOT)
            with tempfile.TemporaryDirectory() as directory:
                raw = Path(directory) / "raw"
                raw.mkdir()
                for name in ["InjuryRecord.csv", "PlayList.csv", "PlayerTrackData.csv"]:
                    (raw / name).touch()
                output = Path(directory) / "run"
                argv = ["main.py", "--raw-dir", str(raw), "--output-dir", str(output)]
                completed = subprocess.CompletedProcess([], 0, stdout="test-package==1\n", stderr="")
                def stage(script, *arguments):
                    if script == "check_paper_results.py":
                        (output / "paper_comparison.json").write_text('{"passed": true}')
                        (output / "SUMMARY.md").write_text("Study summary")

                with patch.object(sys, "argv", argv), patch.object(module.importlib.util, "find_spec", return_value=True), patch.object(module.subprocess, "run", return_value=completed), patch.object(module, "run", side_effect=stage) as called, contextlib.redirect_stdout(io.StringIO()):
                    module.main()
                commands = [call.args for call in called.call_args_list]
                self.assertEqual(commands[0][0], "validate_data.py")
                self.assertEqual(commands[-1][0], "check_paper_results.py")
                self.assertIn("--report-only", commands[-1])
                self.assertEqual(len([c for c in commands if c[0] == "plot_study_results.py"]), 2)
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

    def test_default_run_works_from_another_directory(self):
        module = load("main")
        previous = Path.cwd()
        try:
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                raw = root / "data/raw"
                raw.mkdir(parents=True)
                for name in ["InjuryRecord.csv", "PlayList.csv", "PlayerTrackData.csv"]:
                    (raw / name).touch()
                (root / "splits").mkdir()
                (root / "splits/player_split_assignments.csv").touch()

                def stage(script, *arguments):
                    if script == "check_paper_results.py":
                        report = Path(arguments[arguments.index("--output") + 1])
                        report.write_text('{"passed": false}')
                        (report.parent / "SUMMARY.md").write_text("Study summary")

                os.chdir(root.parent)
                completed = subprocess.CompletedProcess([], 0, stdout="", stderr="")
                messages = io.StringIO()
                with patch.object(module, "ROOT", root), patch.object(sys, "argv", ["main.py"]), patch.object(module.importlib.util, "find_spec", return_value=True), patch.object(module.subprocess, "run", return_value=completed), patch.object(module, "run", side_effect=stage), contextlib.redirect_stdout(messages):
                    module.main()
                    module.main()
                self.assertEqual(len(list((root / "outputs").iterdir())), 2)
                self.assertIn("Paper comparison did not pass", messages.getvalue())
                self.assertNotIn("Paper comparison passed.", messages.getvalue())
        finally:
            os.chdir(previous)

    def test_strict_match_does_not_request_report_only(self):
        module = load("main")
        with tempfile.TemporaryDirectory() as directory:
            raw = Path(directory) / "raw"
            raw.mkdir()
            for name in ["InjuryRecord.csv", "PlayList.csv", "PlayerTrackData.csv"]:
                (raw / name).touch()
            output = Path(directory) / "run"

            def stage(script, *arguments):
                if script == "check_paper_results.py":
                    self.assertNotIn("--report-only", arguments)
                    raise subprocess.CalledProcessError(1, [script])

            completed = subprocess.CompletedProcess([], 0, stdout="", stderr="")
            argv = ["main.py", "--raw-dir", str(raw), "--output-dir", str(output), "--strict-paper-match"]
            with patch.object(sys, "argv", argv), patch.object(module.importlib.util, "find_spec", return_value=True), patch.object(module.subprocess, "run", return_value=completed), patch.object(module, "run", side_effect=stage), contextlib.redirect_stdout(io.StringIO()), self.assertRaises(subprocess.CalledProcessError):
                module.main()

    def test_stage_failure_stops_later_stages(self):
        module = load("main")
        with tempfile.TemporaryDirectory() as directory:
            raw = Path(directory) / "raw"
            raw.mkdir()
            for name in ["InjuryRecord.csv", "PlayList.csv", "PlayerTrackData.csv"]:
                (raw / name).touch()
            argv = ["main.py", "--raw-dir", str(raw), "--output-dir", str(Path(directory) / "run")]
            completed = subprocess.CompletedProcess([], 0, stdout="", stderr="")
            with patch.object(sys, "argv", argv), patch.object(module.importlib.util, "find_spec", return_value=True), patch.object(module.subprocess, "run", return_value=completed), patch.object(module, "run", side_effect=subprocess.CalledProcessError(1, ["validate_data.py"])) as called, contextlib.redirect_stdout(io.StringIO()), self.assertRaises(subprocess.CalledProcessError):
                module.main()
            self.assertEqual(called.call_count, 1)

    def test_child_process_uses_repository_and_current_python(self):
        module = load("main")
        with patch.object(module.subprocess, "run") as called, contextlib.redirect_stdout(io.StringIO()):
            module.run("validate_data.py", "--raw-dir", "data/raw")
        self.assertEqual(called.call_args.args[0][0], sys.executable)
        self.assertEqual(called.call_args.kwargs, {"cwd": ROOT, "check": True})

    def test_comparison_report_only_preserves_failure(self):
        module = load("check_paper_results")
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory) / "comparison.json"
            argv = ["check_paper_results.py", "--primary", "primary", "--sensitivity", "sensitivity", "--output", str(report)]
            failure = [{"item": "test", "passed": False, "expected": 1, "actual": 2}]
            with patch.object(module, "compare", return_value=failure), contextlib.redirect_stdout(io.StringIO()):
                with patch.object(sys, "argv", argv + ["--report-only"]):
                    module.main()
                self.assertFalse(json.loads(report.read_text())["passed"])
                with patch.object(sys, "argv", argv), self.assertRaises(SystemExit):
                    module.main()

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
