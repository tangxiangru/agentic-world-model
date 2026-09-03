#!/usr/bin/env bash
# run_eval.sh <ckpt_dir> <tag> [limit]
# Runs the locked protocol and writes eval/<tag>.json + analysis/<tag>_failure_tags.json
set -euo pipefail
CKPT="$1"; TAG="$2"; LIMIT="${3:-150}"
cd /home/ben/task
python evaluate.py --model-path "$CKPT" --limit "$LIMIT" \
  --max-connections 16 --gpu-memory-utilization 0.85 \
  --json-output-file "eval/${TAG}.json" > "logs/${TAG}_eval.log" 2>&1
python scripts/tag_failures.py "$TAG"
cat "eval/${TAG}.json"
