#!/usr/bin/env bash
# eval_ckpt.sh <checkpoint-path> <tag> [limit]
# Runs the fixed dev protocol: gsm8k --limit N, max-connections 32, gpu-mem 0.85.
set -euo pipefail
CKPT="$1"; TAG="$2"; LIMIT="${3:-200}"
mkdir -p eval logs
python evaluate.py \
  --model-path "$CKPT" \
  --limit "$LIMIT" \
  --max-connections 32 \
  --gpu-memory-utilization 0.85 \
  --json-output-file "eval/${TAG}_dev${LIMIT}.json" \
  > "logs/eval_${TAG}.log" 2>&1
echo "--- $TAG"
cat "eval/${TAG}_dev${LIMIT}.json"
