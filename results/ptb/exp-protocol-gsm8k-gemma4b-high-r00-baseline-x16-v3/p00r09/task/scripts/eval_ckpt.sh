#!/usr/bin/env bash
# Score a checkpoint under the fixed dev protocol (n=200, first 200 gsm8k test
# items) and write both the metric json and a format diagnostic next to it.
set -euo pipefail
CKPT="$1"; TAG="$2"; LIMIT="${3:-200}"
cd /home/ben/task
python evaluate.py --model-path "$CKPT" --limit "$LIMIT" \
  --max-connections 32 --max-tokens 1024 --gpu-memory-utilization 0.6 \
  --json-output-file "eval/${TAG}.json" > "logs/eval_${TAG}.log" 2>&1
LOG=$(ls -t logs/*_gsm8k_*.json | head -1)
python scripts/format_diag.py "$LOG" "analysis/${TAG}_format.json"
echo "== $TAG"; cat "eval/${TAG}.json"
