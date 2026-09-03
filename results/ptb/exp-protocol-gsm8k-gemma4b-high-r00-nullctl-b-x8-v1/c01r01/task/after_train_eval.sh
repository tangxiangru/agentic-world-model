#!/bin/bash
# Wait for the given training pid to exit, then evaluate the resulting checkpoint.
set -u
PID=$1
MODEL=$2
TAG=$3
LIMIT=${4:-200}
while kill -0 "$PID" 2>/dev/null; do sleep 30; done
sleep 20
if [ ! -f "$MODEL/config.json" ]; then
  echo "no checkpoint at $MODEL" >&2
  exit 1
fi
python evaluate.py --model-path "$MODEL" --limit "$LIMIT" --max-connections 32 \
  --gpu-memory-utilization 0.8 --json-output-file "work/${TAG}.json" > "logs/eval_${TAG}.log" 2>&1
echo "=== $TAG ==="
cat "work/${TAG}.json"
