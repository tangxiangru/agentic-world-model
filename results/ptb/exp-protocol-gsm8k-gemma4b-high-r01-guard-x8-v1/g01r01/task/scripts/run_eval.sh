#!/bin/bash
# usage: run_eval.sh <model_path> <tag> [limit]
set -e
M=$1; TAG=$2; LIM=${3:-150}
python evaluate.py --model-path "$M" --limit "$LIM" \
  --json-output-file "eval/${TAG}.json" --max-connections 32 \
  --gpu-memory-utilization 0.85 > "logs/eval_${TAG}.log" 2>&1
echo "--- $TAG"; cat "eval/${TAG}.json"
python scripts/diag.py "$(ls -t logs/*gsm8k*.json | head -1)" | tee "analysis/diag_${TAG}.txt"
cp "$(ls -t logs/*gsm8k*.json | head -1)" "analysis/samples_${TAG}.json"
