#!/bin/bash
# run_eval.sh <tag> <model-path> [limit]
# The protocol in every card: evaluate.py --limit 150 --max-connections 16 --gpu-memory-utilization 0.85
set -euo pipefail
TAG=$1
MODEL=$2
LIMIT=${3:-150}
export HF_HOME=/home/ben/hf_cache
mkdir -p eval logs analysis
python evaluate.py --model-path "$MODEL" --limit "$LIMIT" \
  --max-connections 16 --gpu-memory-utilization 0.85 \
  --json-output-file "eval/${TAG}_dev${LIMIT}.json" > "logs/${TAG}_eval.log" 2>&1
echo "--- eval/${TAG}_dev${LIMIT}.json ---"
cat "eval/${TAG}_dev${LIMIT}.json"
python analyze_eval.py --tag "$TAG"
