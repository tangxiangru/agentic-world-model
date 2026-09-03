#!/usr/bin/env bash
# usage: prep_and_eval.sh <src_ckpt> <tag> [limit]
# Re-saves the checkpoint in bf16 with a greedy generation_config.json, runs the
# grader at --limit, and writes eval/<tag>.json + analysis/<tag>_diag.json.
set -euo pipefail
SRC=$1; TAG=$2; LIMIT=${3:-200}
DST=/home/ben/task/ckpts/${TAG}
cd /home/ben/task
python scripts/to_bf16.py --src "$SRC" --dst "$DST" --greedy 2>&1 | tail -2
python evaluate.py --model-path "$DST" --limit "$LIMIT" --max-connections 32 \
  --gpu-memory-utilization 0.85 --json-output-file "eval/${TAG}.json" \
  > "logs/${TAG}_eval.log" 2>&1
cat "eval/${TAG}.json"
python scripts/analyze_eval.py "$(ls -t logs/*gsm8k*.json | head -1)" "analysis/${TAG}_diag.json"
