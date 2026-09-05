#!/bin/bash
# usage: run_eval.sh <model_path> <tag> [limit] [max_tokens]
set -e
MODEL=$1
TAG=$2
LIMIT=${3:-200}
MAXTOK=${4:-1024}
mkdir -p runs logs/eval_$TAG
INSPECT_LOG_DIR=logs/eval_$TAG timeout 6000 python evaluate.py \
  --model-path "$MODEL" --limit "$LIMIT" --max-connections 32 \
  --max-tokens "$MAXTOK" --gpu-memory-utilization 0.6 \
  --json-output-file "runs/$TAG.json" > "logs/eval_$TAG.log" 2>&1
echo "$TAG:"; cat "runs/$TAG.json"
