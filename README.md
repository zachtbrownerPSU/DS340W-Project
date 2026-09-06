# DS 340W Capstone: NFL Playing Surface and Injury-Risk Prediction

Course: DS 340W, Applied Data Sciences

## Project overview

This project studies whether NFL playing surface contributes useful predictive information about non-contact lower-extremity injuries after accounting for player position, weather, workload, game context, and movement patterns.

The project has two phases:

1. Reproduce the descriptive analysis in the parent paper, including injury frequency, injury severity, injury type, field location, speed, and distance comparisons between synthetic and natural playing surfaces.
2. Extend the parent paper with a reproducible predictive pipeline that uses leakage-safe train, test, and untouched validation groups and compares multiple models.

## Research question

Can current playing-surface and game conditions, combined with player workload and movement information available before an injury, predict a player's risk of a non-contact lower-extremity injury, and does playing surface add predictive value beyond the other factors?

The project evaluates predictive association. The observational dataset cannot by itself establish that synthetic turf causes injuries.

## Papers

### Parent paper

- Kaushal P. Patil and Gregory Kapfhamme, *Machine Learning Meets Sports: Data Analytics Demonstrating the Relationship Between Playing Surface & the Injury/Performance of Athletes* (2022)
- Article: https://www.ijert.org/machine-learning-meets-sports
- PDF: https://www.ijert.org/research/machine-learning-meets-sports-IJERTV11IS080140.pdf

The parent paper uses the NFL playing-surface dataset and reports descriptive comparisons between artificial and natural surfaces. It does not identify a trained predictive model, publish a train/test procedure, or report predictive performance metrics. Our first phase will reproduce its descriptive findings; our main contribution will be the properly validated predictive extension.

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

### Verified local copy

The official ZIP was downloaded from Kaggle and extracted on September 3, 2026. The local project contains:

- `data/raw/InjuryRecord.csv`: 4,928 bytes and 105 injury records
- `data/raw/PlayList.csv`: 24,102,420 bytes and 267,005 player-play records
- `data/raw/PlayerTrackData.csv`: 3,972,378,517 bytes and approximately 76.4 million tracking observations

All three files have the expected names and column headers. Access and redistribution remain subject to the Kaggle competition rules.

## What changes from the parent paper

The parent paper primarily presents exploratory graphs. Our additional contribution is a predictive study with safeguards against inflated performance:

- Use information available before the prediction point rather than information created by the injury outcome.
- Split by `PlayerKey`, not by individual rows, so the same player cannot occur in both training and evaluation data.
- Reserve approximately 10% of players as untouched validation data from the beginning.
- Compare a surface-only baseline against multivariable and nonlinear models.
- Evaluate whether adding playing surface improves out-of-sample predictions.
- Report class-imbalance-aware metrics and uncertainty instead of relying on accuracy alone.

This is important because several public implementations of this dataset randomly split tracking observations or plays. That can leak information from the same player or play into both training and testing and produce unrealistically strong results.

## Proposed unit of analysis and prediction target

The preferred primary unit is the **player-game**:

- Target: whether the player experiences a recorded non-contact lower-extremity injury in the current game.
- Current-game inputs: field surface, weather, stadium type, roster position, and other information known before play.
- Historical inputs: workload and movement summaries calculated only from games or plays before the target game.

A secondary play-level experiment may predict whether a particular play is the recorded injury play using only pre-play or prior-play information. Whole-play tracking measurements must not be described as prospective predictors when they include information recorded after the play started.

## Modeling plan

### Baselines

1. Naive prevalence prediction
2. Surface-only logistic regression
3. Multivariable class-weighted logistic regression

### Enhanced models

4. Class-weighted random forest
5. XGBoost with imbalance-aware weights
6. Optional compact neural network if GPU resources and sample size justify it

### Comparisons

- Static features versus static plus historical workload/movement features
- Best model with field surface versus the same model without field surface
- Model results by surface and, if sample sizes allow, by position group

### Evaluation

- Precision-recall area under the curve (primary ranking metric)
- Recall/sensitivity
- Precision
- F1 score
- ROC-AUC
- Confusion matrix at a stated decision threshold
- Brier score and calibration curve
- Bootstrap confidence intervals where computationally feasible

Accuracy will not be treated as the primary metric because injuries are rare.

## Data split

The target allocation is approximately:

- Training: 70% of players
- Testing: 20% of players
- Final validation: 10% of players

The split will be generated with a fixed random seed and grouped by `PlayerKey`. The validation group will remain untouched until the preprocessing, feature engineering, model selection, and threshold selection steps are complete.

## Existing code references

These projects use the same official dataset and are useful references, but their splitting and evaluation procedures will not be adopted without correction:

- Grand Prize NFL 1st and Future analysis: https://github.com/BenRossJenkins/Grand-Prize-NFL-1st-and-Future-Analytics-Competition
- NFL Injury Analytics: https://github.com/FreshOats/NFL_Injury_Analytics
- Kaggle NFL Injury Analysis notebook: https://www.kaggle.com/code/aleksandradeis/nfl-injury-analysis

## Reproducibility requirements

The final repository should contain:

- A dependency file with fixed or bounded versions
- Scripts that transform the untouched raw files into modeling tables
- Saved train/test/validation player-key manifests
- Model-training and evaluation scripts or notebooks
- Generated figures and metric tables
- Clear instructions that work on a clean machine
- No absolute paths, credentials, private tokens, or restricted raw data

Before the final technical demonstration, the repository must be downloaded onto a computer that has not previously run the project and executed from the README instructions without errors.

## Computing and GPU use

The raw tracking file contains tens of millions of observations. Penn State GPU or ROAR resources may be used for:

- Aggregating and validating the tracking data
- XGBoost hyperparameter searches using GPU acceleration
- Repeated grouped cross-validation and bootstrap evaluation
- SHAP or permutation-importance analysis
- An optional neural-network comparison

The project can retain CPU-compatible baselines so its primary results remain reproducible without specialized hardware.

## Phase 1: reproducing the parent paper's figures

The parent paper (Patil & Kapfhamme, 2022) does not publish its own code. `src/reproduce_paper_figures.py`
rebuilds its eight descriptive figures directly from the raw Kaggle files:

- FIG-1: injury rate (%) by field type
- FIG-2 / FIG-3: injured-player counts and percent-higher-on-synthetic across the 1/7/28/42-day-away buckets
- FIG-4 / FIG-5: injury field-position by body part, artificial vs. natural turf
- FIG-6: injury field-location density, artificial vs. natural turf
- FIG-7 / FIG-8: distribution of per-play max speed and max distance, by surface

### Setup

```
py -m venv .venv
./.venv/Scripts/pip install -r requirements.txt
```

Place the three Kaggle files at `data/raw/InjuryRecord.csv`, `data/raw/PlayList.csv`,
`data/raw/PlayerTrackData.csv` (see `DATA_MANIFEST.md`).

### Run

```
./.venv/Scripts/python src/reproduce_paper_figures.py
```

This reads the raw CSVs (chunked, so the ~4 GB tracking file is never fully loaded into memory)
and writes all eight figures as PNGs to `figures/`. A full run takes under a minute.

## Immediate next steps

1. Reproduce the parent paper's descriptive figures.
2. Generate fixed player-grouped 70/20/10 split manifests.
3. Build the player-game modeling table with historical-only workload and movement features.
4. Train surface-only and multivariable logistic-regression baselines.
5. Train random-forest and XGBoost models.
6. Compare models with and without field surface.
7. Test the final selected pipeline once on the untouched validation players.
8. Package the code and instructions for a clean-machine technical demonstration.

## Academic-integrity boundary

DS 340W permits AI assistance for coding but prohibits AI-generated writing for specified course submissions. This README is a team planning and technical-documentation artifact. Course papers, briefs, presentation text, peer reviews, and other restricted writing must be authored by the students in accordance with the course policy.
