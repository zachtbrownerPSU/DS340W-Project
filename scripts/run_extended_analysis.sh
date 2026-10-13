#!/usr/bin/env bash
set -euo pipefail

source scripts/activate_penn_state_server.sh
mkdir -p outputs/run_logs data/processed
run_id="$(date -u +%Y%m%dT%H%M%SZ)"
export PYTHONHASHSEED=340

date -u > "outputs/run_logs/${run_id}_started_utc.txt"
python --version > "outputs/run_logs/${run_id}_python.txt" 2>&1
python -m pip freeze > "outputs/run_logs/${run_id}_packages.txt"
nvidia-smi > "outputs/run_logs/${run_id}_nvidia_smi.txt"

python scripts/build_play_sequence_table.py \
  --raw-dir data/raw \
  --tracking-features data/processed/play_tracking_features.csv \
  --split-file data/splits/player_split_assignments.csv \
  --output-file data/processed/play_sequence_modeling.csv \
  2>&1 | tee "outputs/run_logs/${run_id}_build_play_sequence.log"

python scripts/run_surface_inference.py \
  --table data/processed/player_game_modeling.csv \
  --output-dir outputs/surface_inference \
  --repetitions 10000 \
  2>&1 | tee "outputs/run_logs/${run_id}_surface_inference.log"

CUDA_VISIBLE_DEVICES=0 python scripts/tune_xgboost.py \
  --table data/processed/player_game_modeling.csv \
  --output-dir outputs/xgboost_search \
  --trials 60 \
  --folds 5 \
  --device cuda \
  2>&1 | tee "outputs/run_logs/${run_id}_xgboost_search.log"

CUDA_VISIBLE_DEVICES=0 python scripts/run_repeated_group_cv.py \
  --table data/processed/player_game_modeling.csv \
  --output-dir outputs/repeated_group_cv \
  --repeats 10 \
  --folds 5 \
  --device cuda \
  2>&1 | tee "outputs/run_logs/${run_id}_repeated_group_cv.log"

CUDA_VISIBLE_DEVICES=1 python scripts/train_play_sequence_models.py \
  --table data/processed/play_sequence_modeling.csv \
  --output-dir outputs/play_sequence_models \
  --windows 5 10 20 \
  --folds 3 \
  --device cuda \
  2>&1 | tee "outputs/run_logs/${run_id}_play_sequence_models.log"

python scripts/analyze_subgroups.py \
  --table data/processed/player_game_modeling.csv \
  --model-dir outputs/model_development_gpu \
  --output-dir outputs/subgroup_analysis \
  2>&1 | tee "outputs/run_logs/${run_id}_subgroup_analysis.log"

python scripts/summarize_extended_results.py \
  --outputs-dir outputs \
  --output-file outputs/EXTENDED_SUMMARY.md \
  2>&1 | tee "outputs/run_logs/${run_id}_summary.log"

date -u > "outputs/run_logs/${run_id}_finished_utc.txt"
echo "extended_analysis=complete run_id=${run_id} validation_used=false"
