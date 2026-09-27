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
