#!/usr/bin/env bash
# eval_ckpt.sh <checkpoint-dir> <tag> [limit] [max-connections]
# Runs the graded protocol and writes eval/<tag>_dev<limit>.json plus the inspect log.
set -euo pipefail
CKPT="$1"; TAG="$2"; LIMIT="${3:-150}"; CONN="${4:-32}"
cd /home/ben/task
export HF_HOME=/home/ben/hf_cache
python evaluate.py \
  --model-path "$CKPT" \
  --limit "$LIMIT" \
  --max-connections "$CONN" \
  --gpu-memory-utilization 0.85 \
  --json-output-file "/home/ben/task/eval/${TAG}_dev${LIMIT}.json" \
  > "/home/ben/task/logs/eval_${TAG}_dev${LIMIT}.log" 2>&1
echo "--- ${TAG} (n=${LIMIT}) ---"
cat "/home/ben/task/eval/${TAG}_dev${LIMIT}.json"
