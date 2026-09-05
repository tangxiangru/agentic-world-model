#!/bin/bash
# usage: bash ev.sh <model_dir> <limit> <tag>
set -e
M=$1; L=${2:-200}; TAG=${3:-$(basename $(dirname $1))_$(basename $1)}
mkdir -p runs logs
python evaluate.py --model-path "$M" --limit "$L" --max-connections 48 \
  --gpu-memory-utilization 0.85 --json-output-file "runs/ev_${TAG}.json" \
  > "logs/ev_${TAG}.log" 2>&1
echo "=== $TAG (limit=$L) ==="
cat "runs/ev_${TAG}.json"
