#!/bin/bash
# run_eval.sh <model-path> <tag> [limit]
# Always the same protocol: --limit 150 (unless overridden), --max-connections 16,
# --gpu-memory-utilization 0.85. Writes eval/<tag>.json and logs/eval_<tag>.log.
set -u
MODEL="$1"; TAG="$2"; LIMIT="${3:-150}"
mkdir -p eval logs
python evaluate.py \
  --model-path "$MODEL" \
  --limit "$LIMIT" \
  --max-connections 16 \
  --gpu-memory-utilization 0.85 \
  --json-output-file "eval/${TAG}.json" > "logs/eval_${TAG}.log" 2>&1
echo "--- eval/${TAG}.json"
cat "eval/${TAG}.json"
