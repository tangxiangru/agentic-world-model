#!/usr/bin/env bash
# Frozen evaluation protocol (exp-01, section 4). Do not change the flags:
# two arms measured under different invocation parameters are not a comparison.
#   usage: scripts/run_eval.sh <model-path> <tag>
set -euo pipefail
MODEL="$1"; TAG="$2"
cd /home/ben/task
rm -rf "logs/$TAG" && mkdir -p "logs/$TAG" analysis eval
INSPECT_LOG_DIR="/home/ben/task/logs/$TAG" python evaluate.py \
  --model-path "$MODEL" \
  --limit 500 \
  --max-connections 2 \
  --gpu-memory-utilization 0.85 \
  --json-output-file "/home/ben/task/eval/${TAG}.json"
python scripts/diag.py --log-dir "/home/ben/task/logs/$TAG" \
  --out "/home/ben/task/analysis/${TAG}_diag.json" \
  --dump-failures "/home/ben/task/analysis/${TAG}_fails.jsonl"
