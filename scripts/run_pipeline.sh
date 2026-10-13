#!/usr/bin/env bash
set -euo pipefail

python scripts/validate_data.py \
  --raw-dir data/raw \
  --output-dir outputs/audit

python scripts/create_player_splits.py \
  --raw-dir data/raw \
  --output-dir data/splits \
  --seed 340

if [[ ! -f data/processed/play_tracking_features.csv ]]; then
  python scripts/build_tracking_features.py \
    --track-file data/raw/PlayerTrackData.csv \
    --output-file data/processed/play_tracking_features.csv
fi

python scripts/build_player_game_table.py \
  --raw-dir data/raw \
  --tracking-features data/processed/play_tracking_features.csv \
  --split-file data/splits/player_split_assignments.csv \
  --output-file data/processed/player_game_modeling.csv

python scripts/reproduce_descriptive_results.py \
  --raw-dir data/raw \
  --output-dir outputs/descriptive

python scripts/train_models.py \
  --table data/processed/player_game_modeling.csv \
  --output-dir outputs/model_development

python scripts/summarize_results.py \
  --model-dir outputs/model_development \
  --repetitions 1000 \
  --seed 340
