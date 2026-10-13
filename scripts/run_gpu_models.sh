#!/usr/bin/env bash
set -euo pipefail

python scripts/train_models.py \
  --table data/processed/player_game_modeling.csv \
  --output-dir outputs/model_development_gpu \
  --include-xgboost \
  --device cuda

python scripts/summarize_results.py \
  --model-dir outputs/model_development_gpu \
  --repetitions 1000 \
  --seed 340
