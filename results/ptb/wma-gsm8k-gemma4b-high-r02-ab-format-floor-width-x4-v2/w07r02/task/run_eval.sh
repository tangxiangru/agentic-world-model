#!/usr/bin/env bash
# The dev protocol, fixed for every card in this session:
#   evaluate.py --limit 150 --max-tokens 4000 --max-connections 32 --gpu-memory-utilization 0.85
# usage: bash run_eval.sh <model-path> <tag>
set -euo pipefail
MODEL="$1"
TAG="$2"
mkdir -p eval logs analysis
python evaluate.py \
  --model-path "$MODEL" \
  --limit 150 \
  --max-tokens 4000 \
  --max-connections 32 \
  --gpu-memory-utilization 0.85 \
  --json-output-file "eval/${TAG}_dev150.json" > "logs/eval_${TAG}.log" 2>&1
LOG=$(grep -o 'logs/[0-9T:+-]*_gsm8k_[A-Za-z0-9]*\.json' "logs/eval_${TAG}.log" | tail -1)
echo "== $TAG =="
cat "eval/${TAG}_dev150.json"
python analyze_eval.py --log "$LOG" --out "analysis/${TAG}_diag.json" | head -25
