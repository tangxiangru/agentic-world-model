#!/bin/bash
# Usage: bash run_eval.sh <model_dir> <limit> <tag>
set -e
MODEL="$1"; LIMIT="${2:-100}"; TAG="${3:-eval}"
python evaluate.py --model-path "$MODEL" --limit "$LIMIT" \
  --max-connections 6 --json-output-file "metrics_${TAG}.json" > "eval_${TAG}.log" 2>&1
echo "=== metrics_${TAG}.json ==="
cat "metrics_${TAG}.json"
