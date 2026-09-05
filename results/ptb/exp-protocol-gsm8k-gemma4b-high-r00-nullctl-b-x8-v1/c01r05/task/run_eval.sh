#!/bin/bash
# usage: run_eval.sh <model_path> <tag> [limit] [gpu_util]
set -u
MODEL=$1
TAG=$2
LIMIT=${3:-150}
GPU=${4:-0.55}
mkdir -p runs logs
python evaluate.py --model-path "$MODEL" --limit "$LIMIT" \
  --max-connections 32 --gpu-memory-utilization "$GPU" \
  --json-output-file "runs/${TAG}.json" > "logs/eval_${TAG}.log" 2>&1
echo "== $TAG =="
cat "runs/${TAG}.json"
