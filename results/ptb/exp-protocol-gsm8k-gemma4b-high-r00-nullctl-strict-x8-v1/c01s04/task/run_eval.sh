#!/bin/bash
# usage: run_eval.sh <model_path> <tag> [limit] [extra args]
set -u
MODEL=$1
TAG=$2
LIMIT=${3:-150}
shift 3 2>/dev/null || shift 2
python evaluate.py --model-path "$MODEL" --limit "$LIMIT" \
  --max-connections 32 --gpu-memory-utilization 0.85 \
  --json-output-file "logs/${TAG}.json" "$@" > "logs/${TAG}.log" 2>&1
echo "=== $TAG ==="
cat "logs/${TAG}.json"
