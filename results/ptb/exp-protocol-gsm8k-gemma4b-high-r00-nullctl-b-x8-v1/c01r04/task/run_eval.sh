#!/bin/bash
# usage: bash run_eval.sh <model_path> <tag> [limit] [gpu_util]
set -e
M=$1; TAG=$2; LIM=${3:-200}; GU=${4:-0.85}
python evaluate.py --model-path "$M" --limit "$LIM" --max-connections 32 --max-tokens 1024 \
  --gpu-memory-utilization "$GU" --json-output-file "logs/eval_${TAG}.json" > "logs/eval_${TAG}.log" 2>&1
echo "== $TAG =="
cat "logs/eval_${TAG}.json"
