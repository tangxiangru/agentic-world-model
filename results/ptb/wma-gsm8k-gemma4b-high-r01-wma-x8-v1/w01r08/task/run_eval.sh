#!/bin/bash
# usage: run_eval.sh <ckpt> <tag> [limit]
set -e
CKPT=$1; TAG=$2; LIM=${3:-150}
BEFORE=$(ls logs/*gsm8k*.json 2>/dev/null | wc -l)
python evaluate.py --model-path "$CKPT" --limit "$LIM" --max-connections 16 \
  --gpu-memory-utilization 0.6 --json-output-file "eval/${TAG}.json" > "logs/eval_${TAG}.log" 2>&1
NEW=$(ls -t logs/*gsm8k*.json | head -1)
cp "$NEW" "logs/inspect_${TAG}.json"
python analyze_log.py "logs/inspect_${TAG}.json" "analysis/wrong_${TAG}.jsonl" > "analysis/summary_${TAG}.json"
cat "analysis/summary_${TAG}.json"
