#!/bin/bash
# usage: run_eval.sh <model_path> <tag> [limit] [extra args]
set -u
MODEL=$1; TAG=$2; LIMIT=${3:-200}
export HF_HOME=/home/ben/hf_cache
export INSPECT_LOG_DIR=logs/inspect_$TAG
mkdir -p runs
python evaluate.py --model-path "$MODEL" --limit "$LIMIT" \
  --max-connections 32 --gpu-memory-utilization 0.85 \
  --json-output-file "runs/$TAG.json" > "logs/eval_$TAG.log" 2>&1
echo "=== $TAG ==="; cat "runs/$TAG.json"
