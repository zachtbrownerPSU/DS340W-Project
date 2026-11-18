# Reproduce the feature-group study

Use a computer that has not run this project for the class recording. Download the repository during the recorded session. Running SSH from a new laptop still executes on the old server, so do not use that as the fresh-machine demonstration without the instructor's approval.

## Before recording

Both team members need to appear on webcam. Start the Zoom recording and screen share before downloading the repository. Rehearse the instructions in a separate clean environment. Keep the final recording computer unused for this project until the recording unless the instructor permits a rehearsal there. Check that software installation and a download of roughly 4 GB are permitted. Keep enough free space for the raw files, extracted tracking data, processed tables, and a new output directory. Do not promise a runtime without a relevant measured run.

The model settings are unchanged. The full command rebuilds tracking features from the raw files, then fits 600 models across the primary and sensitivity analyses. It also runs 2,000 bootstrap draws for each analysis. This is not a short display of previously saved results.

## Download and setup

Download the ZIP from https://github.com/zachtbrownerPSU/DS340W-Project using Code, then Download ZIP, and extract it. Open a terminal in the extracted project directory.

For the closest match to the recorded study, use Python 3.14.4 and the pinned study requirements below. Check the version with `python --version` or the environment's executable. A different Python or package combination has not been shown to reproduce the paper values.

Windows PowerShell:

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements-study.txt
```

macOS or Linux:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements-study.txt
```

These commands call the environment's Python directly and do not require changing PowerShell execution policy. A managed lab computer may prevent installing Python or packages. Arrange a permitted environment before recording.

The historical run manifest records Python 3.14.4, NumPy 2.5.3, pandas 3.0.6, scikit-learn 1.9.1, and XGBoost 3.4.1. The ordinary requirements file is a compatibility range, not a lockfile. For a closer match, install `requirements-study.txt` instead. It pins the model and plotting packages and the recorded supporting dependencies, including NCCL on Linux x86-64. It does not pin Python, the operating system, drivers, or every platform-specific dependency. Check package availability for the recording computer. The new run saves `packages.txt` for its complete installed environment.

## Data

Sign in to Kaggle, complete its access requirements yourself, and download the official competition files:

https://www.kaggle.com/competitions/nfl-playing-surface-analytics/data

Place `InjuryRecord.csv`, `PlayList.csv`, and `PlayerTrackData.csv` in `data/raw`. Do not rename columns or substitute a different dataset. The repository contains the original player split assignments under `splits`; the reproduction command uses those exact assignments rather than creating new ones.

The injury file contains 105 records but 104 distinct injury player-games. Descriptive severity summaries count records. The classifier counts whether each player-game contains an injury. These totals are intentionally different.

## Run

Recording warning from the November 18 rehearsal: the full NVIDIA run on the original server reproduced the study, but a CPU XGBoost check did not give the same four-decimal AP. In primary repeat 1, fold 1, context-only XGBoost gave AP 0.02697 on CPU versus 0.02644 in the recorded GPU fit. This is a fold-level diagnostic, not a new pooled study result. CPU execution is supported, but the GPU paper reference is not a verified CPU reference. The commands below can run on CPU, yet the final paper comparison may fail for that reason. Do not record a CPU-only machine as ready until its complete results are checked and any CPU-specific reporting is agreed. Do not increase the tolerance to conceal the difference.

Windows PowerShell:

```powershell
.\.venv\Scripts\python.exe scripts/reproduce_project.py --output-dir outputs/recording_run --device cpu --threads 4
```

macOS or Linux:

```bash
.venv/bin/python scripts/reproduce_project.py --output-dir outputs/recording_run --device cpu --threads 4
```

On a compatible NVIDIA computer, `--device cuda:0` trains XGBoost on the GPU. Logistic regression and random forest remain CPU models. CPU and GPU fits may differ numerically; changing the device is not guaranteed to reproduce every printed digit.

Use a new output directory for every run. The command will not overwrite an existing run. If it fails, retain the output and read the first error. Do not skip a failed stage and present the later files as a successful full reproduction.

## Show the results

Open these newly generated files after the run:

- `outputs/recording_run/audit/data_audit.json`: raw data counts and the parent-paper discrepancy.
- `outputs/recording_run/primary/data_audit.json`: 209 players, 4,776 player-games, 90 injury games.
- `outputs/recording_run/sensitivity/data_audit.json`: 225 players, 5,192 player-games, 93 injury games.
- `outputs/recording_run/primary/SUMMARY.md`: primary model metrics and feature comparisons.
- `outputs/recording_run/sensitivity/SUMMARY.md`: sensitivity metrics.
- `outputs/recording_run/primary/roc_precision_recall.png`: saved ROC and precision-recall curves.
- `outputs/recording_run/primary/verification.json` and the sensitivity equivalent: recomputation of metrics from the new predictions and fold checks.
- `outputs/recording_run/paper_comparison.json`: comparison with the recorded study values.

Table I reports mean complete out-of-fold metrics across five repeats. Table II uses predictions averaged across repeats before calculating AP differences. The two summaries need not have identical differences. The bootstrap intervals apply to the second summary, not the difference between Table I means.

Create the paper's column-sized black-and-white figures from a completed run:

```bash
python scripts/plot_study_results.py --run-dir outputs/recording_run/primary --output-dir outputs/recording_run/paper_figures
```

This fits no models. It saves cohort flow, logistic ROC and precision-recall curves, and the paired AP interval plot as PNG and PDF files, plus the averaged-prediction curve metrics as CSV. The precision axis is limited to 0-0.22 for readability, as stated in the paper caption. The ROC and PR legend values use probabilities averaged across repeats, not Table I's mean of repeat metrics.

`check_paper_results.py` checks the saved model metrics, cohort counts, and paired AP intervals against `reference/paper_results.json`. Its absolute tolerance is 0.00005, appropriate to four-decimal reporting. It is not a claim of bitwise equality. A failed comparison must be investigated, not hidden by increasing the tolerance. This comparison is separate from the verifier that recalculates metrics from prediction files.

## What each person should explain

Agree on the speaking split before recording. One person can explain raw-file checks, tracking aggregation, the outcome, player splits, and history construction. The other can explain the four input groups, models, shared folds, bootstrap, and output verification. Both should explain code they understand and actually worked on; adjust these suggestions to your real contributions.

Show why the 16-player chronology conflict changes the analysis, why validation players are excluded from the follow-up, and why uncertain AP differences are not proof that surface has no effect. Finish with the comparison report and the actual tables used in the paper.

## Earlier and external analyses

The full reproduction command targets the paper's main controlled feature-group study. It does not reproduce every earlier exploratory experiment mentioned in the discussion.

To reproduce the already selected earlier random forest, without selecting a different model, run:

```bash
python scripts/train_models.py --table outputs/recording_run/processed/player_game_modeling.csv --output-dir outputs/earlier_validation_reproduction --final-model full_random_forest_no_surface
python scripts/run_surface_inference.py --table outputs/recording_run/processed/player_game_modeling.csv --output-dir outputs/surface_reproduction --repetitions 10000 --seed 340
```

Use the environment-specific Python executable above in place of `python` if the environment is not activated. The first command is reproduction of an already viewed evaluation, not a new untouched test. These historical outputs require their own comparison with the original server files. The original extended scripts also cover the XGBoost search, play-window experiments, and subgroup checks; they are server-oriented and substantially longer.

The external nflverse cohort has a different report-based target. It is not added to the Kaggle injury labels and is not claimed as external model validation. Its separate download and construction commands remain in the main README. Ask the instructor which historical or secondary analyses must be included in the recording rather than assuming that a shorter subset satisfies the assignment.

## Status

Both November 18 server rehearsals completed successfully from the raw data: one in the established environment and one in a separately installed virtual environment. Each fitted all 600 models, completed both 2,000-draw bootstraps, passed prediction verification, and passed the paper comparison. All saved repeat AP, ROC-AUC, and Brier scores matched the original study exactly. This verifies the Linux NVIDIA route on this server, not an unused computer or CPU equivalence.

Run the standard-library checks with `python -m unittest discover -s tests -v`. These mock model execution and do not test a real fit. A complete unused-machine run must still be executed and compared before the recording is described as meeting that requirement.

## Manual run on the classroom server

This is a tested server workflow, not proof of compliance with the unused-machine assignment. Obtain instructor approval if you intend to reuse this server for that assignment. The commands reuse the clean environment and raw data already present in the team's shared directory. They do not install a new machine from scratch.

Connect to the Penn State VPN and SSH into the GPU server. For an approved recording, start Zoom, screen sharing, and both webcams before the GitHub download. Choose a new clone directory if the one below already exists; never delete another run to reuse its name.

```bash
cd /data/ds340w-003-team1
git clone https://github.com/zachtbrownerPSU/DS340W-Project.git DS340W-recording
cd DS340W-recording
source /data/ds340w-003-team1/reproduction_20261118/.venv-clean/bin/activate
python --version
python -m pip check
nvidia-smi -L
python scripts/reproduce_project.py --raw-dir /data/ds340w-003-team1/data/raw --output-dir outputs/recording_run --device cuda:0 --threads 8
```

Wait for `Paper comparison: PASS` and the final completion message. A returned prompt alone is not evidence of success. Do not continue past a traceback as though the run completed.

```bash
python scripts/plot_study_results.py --run-dir outputs/recording_run/primary --output-dir outputs/recording_run/paper_figures
cat outputs/recording_run/SUMMARY.md
cat outputs/recording_run/primary/verification.json
cat outputs/recording_run/sensitivity/verification.json
python -c "import json; r=json.load(open('outputs/recording_run/paper_comparison.json')); print('Paper comparison passed:', r['passed'])"
```

These commands generate new results, not a display of the November 15 files. The main run covers the feature-group experiment, not every older exploratory experiment mentioned in the paper. Runtime on a shared server can change with other users' jobs.
