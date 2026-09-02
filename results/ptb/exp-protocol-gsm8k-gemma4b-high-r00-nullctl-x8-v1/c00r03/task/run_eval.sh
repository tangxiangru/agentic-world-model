#!/bin/bash
# usage: run_eval.sh <model_path> <tag> [limit] [max_tokens]
set -e
MP=$1; TAG=$2; LIM=${3:-200}; MT=${4:-1024}
mkdir -p runs
python evaluate.py --model-path "$MP" --limit "$LIM" --max-connections 32 \
  --gpu-memory-utilization 0.85 --max-tokens "$MT" \
  --json-output-file "runs/${TAG}.json" > "runs/${TAG}.log" 2>&1
echo "=== $TAG ==="; cat "runs/${TAG}.json"
