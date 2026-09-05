#!/usr/bin/env bash
# Score a checkpoint under the standing protocol of this run:
#   evaluate.py --limit 150 --max-connections 16 --gpu-memory-utilization 0.6
# usage: bash run_eval.sh <checkpoint-dir> <tag> [limit]
set -euo pipefail
CKPT="$1"; TAG="$2"; LIMIT="${3:-150}"
mkdir -p eval logs/inspect_"$TAG"
INSPECT_LOG_DIR=/home/ben/task/logs/inspect_"$TAG" \
python evaluate.py --model-path "$CKPT" --limit "$LIMIT" \
  --max-connections 16 --gpu-memory-utilization 0.6 \
  --json-output-file /home/ben/task/eval/"$TAG".json \
  > logs/eval_"$TAG".log 2>&1
python analyze_eval.py logs/inspect_"$TAG" analysis/"$TAG"_diagnostic.json
cat eval/"$TAG".json
