# DS 340W Capstone: NFL Playing Surface and Injury-Risk Prediction

Course: DS 340W, Applied Data Sciences

## Project overview

This project studies whether NFL playing surface helps predict recorded lower-limb injuries after accounting for player position, weather, workload, game context, and movement patterns. The Kaggle labels do not establish whether every injury was non-contact.

The project has two phases:

1. Reproduce the descriptive analysis in the parent paper, including injury frequency, injury severity, injury type, field location, speed, and distance comparisons between synthetic and natural playing surfaces.
2. Extend the parent paper with a predictive pipeline that separates players across development and validation groups and compares multiple models.

## Research question

Does playing surface add predictive value for a recorded lower-limb injury during a player-game beyond game context and workload from earlier games?

The project evaluates predictive association. The observational dataset cannot by itself establish that synthetic turf causes injuries.

## Papers

### Parent paper

- Kaushal P. Patil and Gregory Kapfhamme, *Machine Learning Meets Sports: Data Analytics Demonstrating the Relationship Between Playing Surface & the Injury/Performance of Athletes* (2022)
- Article: https://www.ijert.org/machine-learning-meets-sports
- PDF: https://www.ijert.org/research/machine-learning-meets-sports-IJERTV11IS080140.pdf

The parent paper reports descriptive comparisons between artificial and natural surfaces. It does not identify a trained predictive model, publish a train/test procedure, or report predictive performance metrics. We use the Kaggle dataset to reproduce its approach, but its exact source files remain unconfirmed and the reported injury totals differ from the Kaggle files. This prevents a claim of exact numerical replication.

### Supporting papers

1. Nikit Venishetty et al., *Lower Extremity Injury Rates on Artificial Turf Versus Natural Grass Surfaces in the National Football League During the 2021 and 2022 Seasons* (2024)
   - https://journals.sagepub.com/doi/10.1177/23259671241265378
   - https://journals.sagepub.com/doi/pdf/10.1177/23259671241265378?download=true

2. William F. McCormick et al., *Field Surface Type and Season-Ending Lower Extremity Injury in NFL Players* (2024)
   - https://onlinelibrary.wiley.com/doi/full/10.1155/2024/6832213
   - https://doi.org/10.1155/2024/6832213

The supporting studies reach different conclusions after accounting for exposure. This disagreement motivates an exposure-aware predictive analysis instead of assuming in advance that one surface is inherently more dangerous.

## Official dataset

- NFL 1st and Future Analytics competition: https://www.kaggle.com/competitions/nfl-playing-surface-analytics
- Data page: https://www.kaggle.com/competitions/nfl-playing-surface-analytics/data
- License: subject to the Kaggle competition rules
- Download size: approximately 4 GB
- Population: complete in-game histories for 250 players across two NFL regular seasons
- Injuries: 105 recorded lower-limb injuries; some injuries do not have a known `PlayKey`

### Files

- `InjuryRecord.csv`: injury location, playing surface, and days-missed indicators
- `PlayList.csv`: 267,005 player-play records with position, stadium, surface, weather, play type, and timing information
- `PlayerTrackData.csv`: location, orientation, speed, direction, and distance observations recorded at 10 Hz

Raw competition files belong under `data/raw/` and should not be committed to a public repository unless the competition license permits redistribution. Reproducible download and preprocessing instructions should be committed instead.

## Run the modeling code

Download or clone this repository and open a terminal in its folder. Install Python 3.11 or newer. The original study used Python 3.14.4. The code runs on Windows, macOS, or Linux and defaults to CPU; an NVIDIA GPU is optional. A complete run processes the large tracking file and fits 600 models, so allow substantial time and free disk space. A small laptop will be slower than a workstation.

### Install once

Windows PowerShell:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

If PowerShell blocks activation, use `.\.venv\Scripts\python.exe` instead of `python` for installation and running the project. You do not need to change the computer's execution policy.

macOS or Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

For the original study's exact package versions, use `requirements-study.txt` instead of `requirements.txt`. Package availability depends on Python version and platform. The pinned file describes the original environment, not a guarantee of identical numbers on every computer.

### Add the data

Download the official competition files from [Kaggle](https://www.kaggle.com/competitions/nfl-playing-surface-analytics/data), complete its access requirements, and extract the CSV files into this layout:

```text
DS340W-Project/
    main.py
    data/
        raw/
            InjuryRecord.csv
            PlayList.csv
            PlayerTrackData.csv
```

The injury file contains 105 records but 104 distinct injury player-games. Severity summaries count injury records; the classifier predicts whether a player-game contains an injury. The existing player assignments in `splits/player_split_assignments.csv` must stay unchanged.

### Run everything

```text
python main.py
```

There is no need to run the individual scripts or use Bash. `main.py` runs the complete main study in order:

1. Check the raw data and save the data audit.
2. Aggregate tracking observations into play-level movement features.
3. Build the player-game table and earlier-game history using the committed splits.
4. Recreate the descriptive surface comparisons.
5. Run the primary and sensitivity feature-group analyses, including prediction verification.
6. Summarize the metrics and paired player-bootstrap comparisons.
7. Generate ROC, precision-recall, cohort-flow, and paired-interval figures for both analyses.
8. Compare the results with the reference values and print the study summary.

The study compares context, context plus surface, context plus earlier workload, and all inputs together. Logistic regression, random forest, and XGBoost share player-grouped folds. Five repeats of five folds across both timing policies produce 600 fits, followed by 2,000 player-bootstrap draws per analysis. These settings do not shrink on CPU.

For a compatible NVIDIA GPU with a working driver and GPU-enabled XGBoost installation:

```text
python main.py --device cuda:0
```

Only XGBoost uses the GPU. The other models and most data processing use the CPU. Optional arguments are shown by `python main.py --help`; these include data location, output location, thread count, and tracking chunk size. Relative paths are resolved from the repository folder. Every default run gets a new timestamped output folder. Existing output folders are never overwritten.

### Read the results

The terminal prints the new run folder, for example `outputs/run_20261120_140000_123456/`. Open these files inside it:

- `SUMMARY.md`: combined primary and sensitivity findings.
- `audit/`: raw-data checks and descriptive source discrepancies.
- `descriptive/`: descriptive comparisons and figures.
- `primary/SUMMARY.md` and `sensitivity/SUMMARY.md`: model metrics and interpretation.
- `primary/figures/` and `sensitivity/figures/`: paper-style PNG and PDF figures.
- `primary/verification.json` and `sensitivity/verification.json`: prediction and fold checks.
- `paper_comparison.json`: reference values, observed values, tolerance, and pass/fail checks.
- `packages.txt` and `started.json`: environment and run settings.

The primary analysis excludes 16 players with conflicting game-day order. It contains 209 players, 4,776 player-games, and 90 injury games. The sensitivity analysis retains those players under a listed-game-order assumption, giving 225 players, 5,192 player-games, and 93 injury games. See [the method description](docs/feature_ablation.md) for details.

CPU and GPU XGBoost can give different scores. Different package versions can also change results. The paper comparison retains its absolute tolerance of 0.00005 and always reports mismatches. By default a reference mismatch does not discard an otherwise completed analysis or hide its results. It is not a successful reproduction of the paper values. To require a matching reference as well as a completed analysis, run:

```text
python main.py --strict-paper-match
```

Any failed processing, training, prediction-verification, or plotting stage stops the pipeline. Read the first error before retrying. The main study has been fully rerun on Linux with NVIDIA hardware; a complete fresh Windows or macOS run has not yet been verified.

## Scope and limitations

The original validation results were examined before this follow-up. The feature-group study uses the original training and testing players only. That earlier validation result is not an untouched test for later changes. The dataset does not establish non-contact mechanism or causal effects of playing surface.

`main.py` runs the complete controlled feature-group study and descriptive analysis. Older tuning, play-window, subgroup, and external-cohort experiments remain separate modules; they are not silently added to this study or its injury labels. The external cohort uses public NFL injury reports and has a different outcome definition.

Raw data, processed data, fitted models, and generated results are excluded from GitHub. The repository contains code and documentation needed to understand and run it.

## Tests

```text
python -m unittest discover -s tests -v
```

The runner tests mock expensive training stages. They check orchestration and failure handling, not the numerical accuracy of a newly trained model.
