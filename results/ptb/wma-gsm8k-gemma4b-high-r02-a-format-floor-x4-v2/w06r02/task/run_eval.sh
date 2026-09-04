#!/usr/bin/env bash
# run_eval.sh <model-path> <tag> [limit]
# The dev protocol, fixed since exp-01: --limit 150 --max-connections 16 --gpu-memory-utilization 0.6
set -euo pipefail
MODEL="$1"; TAG="$2"; LIMIT="${3:-150}"
mkdir -p eval logs analysis
python evaluate.py --model-path "$MODEL" --limit "$LIMIT" --max-connections 16 \
  --gpu-memory-utilization 0.6 --json-output-file "eval/${TAG}.json" > "logs/eval_${TAG}.log" 2>&1
LOG=$(ls -t logs/*_gsm8k_*.json | head -1)
cp "$LOG" "eval/${TAG}_log.json"
python analyze_log.py "eval/${TAG}_log.json" --dump-failures "analysis/${TAG}_failures.jsonl" | tee "analysis/${TAG}_diag.json"
