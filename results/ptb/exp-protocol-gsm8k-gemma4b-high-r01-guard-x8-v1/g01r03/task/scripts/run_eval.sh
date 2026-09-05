#!/usr/bin/env bash
# Dev protocol P1: gsm8k test --limit N through the harness's own evaluate.py.
# usage: scripts/run_eval.sh <model-path> <tag> [limit]
set -euo pipefail
MODEL="$1"; TAG="$2"; LIMIT="${3:-300}"
cd /home/ben/task
mkdir -p eval logs analysis
python evaluate.py \
  --model-path "$MODEL" \
  --limit "$LIMIT" \
  --max-connections 32 \
  --gpu-memory-utilization 0.8 \
  --json-output-file "eval/${TAG}.json" \
  > "logs/eval_${TAG}.log" 2>&1
echo "--- ${TAG} ---"
cat "eval/${TAG}.json"
