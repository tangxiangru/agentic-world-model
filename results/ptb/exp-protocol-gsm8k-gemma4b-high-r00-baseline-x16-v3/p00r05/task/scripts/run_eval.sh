#!/usr/bin/env bash
# The one evaluation protocol every card in this batch uses.
#   run_eval.sh <model-path> <tag> [limit]
# Writes eval/<tag>.json (metrics) and logs/eval_<tag>.log; the inspect_ai
# per-sample log lands in logs/<timestamp>_gsm8k_*.json.
set -euo pipefail
MODEL="$1"; TAG="$2"; LIMIT="${3:-150}"
cd /home/ben/task
mkdir -p eval logs "logs/$TAG"
INSPECT_LOG_DIR="logs/$TAG" python evaluate.py \
  --model-path "$MODEL" \
  --limit "$LIMIT" \
  --max-connections 32 \
  --gpu-memory-utilization 0.85 \
  --json-output-file "eval/${TAG}.json" > "logs/eval_${TAG}.log" 2>&1
echo "--- eval/${TAG}.json"
cat "eval/${TAG}.json"
