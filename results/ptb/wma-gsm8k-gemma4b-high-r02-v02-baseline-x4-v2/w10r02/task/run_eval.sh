#!/bin/bash
# usage: run_eval.sh <ckpt_dir> <tag> [limit]
set -e
CKPT=$1; TAG=$2; LIMIT=${3:-150}
mkdir -p eval/logs/$TAG
INSPECT_LOG_DIR=/home/ben/task/eval/logs/$TAG python evaluate.py \
  --model-path $CKPT --limit $LIMIT \
  --json-output-file /home/ben/task/eval/${TAG}.json \
  --max-connections 16 --max-tokens 4000 --gpu-memory-utilization 0.3 \
  > logs/eval_${TAG}.log 2>&1
cat eval/${TAG}.json
python analyze_log.py eval/logs/$TAG --dump-failures analysis/${TAG}_failures.json
