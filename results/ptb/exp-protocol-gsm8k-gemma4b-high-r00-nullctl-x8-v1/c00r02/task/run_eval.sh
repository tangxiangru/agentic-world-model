#!/usr/bin/env bash
# usage: bash run_eval.sh <model_dir> <limit> <tag> [max_connections]
set -u
MODEL=$1
LIMIT=${2:-150}
TAG=${3:-eval}
MC=${4:-32}
mkdir -p logs
python evaluate.py --model-path "$MODEL" --limit "$LIMIT" \
  --max-connections "$MC" --gpu-memory-utilization 0.85 \
  --json-output-file "logs/${TAG}.json" > "logs/${TAG}.log" 2>&1
echo "=== $TAG ==="
cat "logs/${TAG}.json" 2>/dev/null
tr '\r' '\n' < "logs/${TAG}.log" | grep -viE "POST /v1" | tail -12
