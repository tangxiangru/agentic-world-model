#!/usr/bin/env bash
# Dev evaluation protocol. Fixed for every card so numbers are comparable.
#   run_eval.sh <model_path> <tag> [limit]
set -euo pipefail
MODEL="$1"
TAG="$2"
LIMIT="${3:-200}"
cd /home/ben/task
mkdir -p eval logs
export INSPECT_LOG_DIR="/home/ben/task/eval/logs_${TAG}"
python evaluate.py \
  --model-path "$MODEL" \
  --limit "$LIMIT" \
  --max-connections 32 \
  --max-tokens 1024 \
  --gpu-memory-utilization 0.85 \
  --json-output-file "eval/${TAG}.json" 2>&1 | tail -40
echo "=== ${TAG} ==="
cat "eval/${TAG}.json"
