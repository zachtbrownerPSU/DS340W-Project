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

For the fresh-machine recording, follow [the reproduction instructions](docs/reproduce.md). They include Windows and macOS setup and one command that rebuilds the main study from the raw data. It uses the committed player assignments, keeps the original experiment settings, and checks the new metrics against the recorded study. The new CPU path does not require `nvidia-smi`. A full new-machine run must still be verified before recording.

Install the Python packages:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Place the three Kaggle files listed above under `data/raw/`, then run the main pipeline:

```bash
bash scripts/run_pipeline.sh
```

On the Penn State classroom GPU server, use the project-local package setup because the server does not provide `ensurepip`:

```bash
bash scripts/setup_penn_state_server.sh
source scripts/activate_penn_state_server.sh
bash scripts/run_pipeline.sh
bash scripts/run_gpu_models.sh
```

The extended experiments use repeated player-grouped cross-validation, GPU XGBoost tuning, surface inference, subgroup checks, and 5-play, 10-play, and 20-play prediction windows:

```bash
bash scripts/run_extended_analysis.sh
```

The original validation results have already been examined. Follow-up experiments use the original train and test players only and are exploratory; the earlier validation result is not an untouched test for later model changes. Generated data, fitted models, logs, and outputs are excluded from the repository.

## Compare surface and earlier workload

The feature ablation study compares context, context plus surface, context plus earlier workload, and all features together. Logistic regression, random forest, and XGBoost use identical player-grouped folds within each run.

```bash
source scripts/activate_penn_state_server.sh
python scripts/run_feature_ablation.py --table data/processed/player_game_modeling.csv --output-dir outputs/feature_ablation_primary --device cuda:0 --chronology-policy exclude_players
python scripts/run_feature_ablation.py --table data/processed/player_game_modeling.csv --output-dir outputs/feature_ablation_sensitivity --device cuda:1 --chronology-policy game_order
python scripts/summarize_feature_ablation.py --primary outputs/feature_ablation_primary --sensitivity outputs/feature_ablation_sensitivity --output outputs/feature_ablation_summary.md
```

The primary run excludes players whose listed game days go backward. The sensitivity run keeps them under an explicit game-order assumption and treats nonpositive rest intervals as missing. Existing tables and results are preserved. See [feature ablation instructions](docs/feature_ablation.md) for the timing audit, feature definitions, evaluation, and output files.

## Build the external cohort

The external analysis uses public 2020 through 2024 NFL injury reports, weekly rosters, snap counts, schedules, and playing-surface information. It remains separate from the anonymous Kaggle cohort.

```bash
bash scripts/download_external_nfl_data.sh
python scripts/build_external_injury_cohort.py
```

The player-game cohort and game-level surface table are written under `data/external/processed/` and are excluded from GitHub because they can be rebuilt from the source data.
