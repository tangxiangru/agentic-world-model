#!/bin/bash
# usage: bash run_eval.sh <model_path> <limit> <tag>
MP="$1"; LIM="${2:-150}"; TAG="${3:-eval}"
cd /home/ben/task
python evaluate.py --model-path "$MP" --limit "$LIM" \
  --json-output-file "logs/${TAG}.json" --max-connections 4 --max-tokens 1024 \
  > "logs/${TAG}.log" 2>&1
echo "=== $TAG ==="; cat "logs/${TAG}.json"
