# Dataset access and local manifest

The raw files for this project come from the Kaggle competition [NFL 1st and Future Analytics](https://www.kaggle.com/competitions/nfl-playing-surface-analytics/data).

Each team member must sign in to Kaggle and accept the competition rules before accessing the raw data. The files may be used for academic research and education under those rules. Do not publish the raw files or give them to anyone who has not accepted the rules.

## Verified files

The local copy was downloaded and checked on September 3, 2026.

| File | Size in bytes | Records | Purpose |
| --- | ---: | ---: | --- |
| `raw/InjuryRecord.csv` | 4,928 | 105 | Injury location, surface, and days-missed indicators |
| `raw/PlayList.csv` | 24,102,420 | 267,005 | Player-play context, field, weather, position, and workload fields |
| `raw/PlayerTrackData.csv` | 3,972,378,517 | 76,366,748 | Player tracking observations recorded at 10 Hz |

The files were extracted from `nfl-playing-surface-analytics.zip`. Their names and column headers match the Kaggle data description.

## Expected location

After downloading and extracting the Kaggle ZIP, place the files here:

```text
data/
  raw/
    InjuryRecord.csv
    PlayList.csv
    PlayerTrackData.csv
```
