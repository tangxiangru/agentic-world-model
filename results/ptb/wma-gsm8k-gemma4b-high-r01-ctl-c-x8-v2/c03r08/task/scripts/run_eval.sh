#!/usr/bin/env bash
# usage: run_eval.sh <model_path> <tag> [limit] [max_connections]
set -euo pipefail
MODEL="$1"; TAG="$2"; LIMIT="${3:-150}"; MC="${4:-16}"
cd /home/ben/task
mkdir -p eval logs
python evaluate.py --model-path "$MODEL" --limit "$LIMIT" \
  --json-output-file "eval/${TAG}.json" \
  --max-connections "$MC" --max-tokens 4000 --gpu-memory-utilization 0.85 \
  > "logs/eval_${TAG}.log" 2>&1
echo "--- $TAG ---"; cat "eval/${TAG}.json"
