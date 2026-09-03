#!/usr/bin/env bash
# Stage a checkpoint into a directory vLLM can load, then run the grader's own
# protocol on it. Usage: run_eval.sh <ckpt_dir> <tag> [decode: inherit|greedy] [limit]
set -euo pipefail
CKPT="$1"; TAG="$2"; DECODE="${3:-inherit}"; LIMIT="${4:-150}"
STAGE="/home/ben/task/ckpts/_stage_${TAG}"
rm -rf "$STAGE"
python /home/ben/task/scripts/package_final.py --ckpt "$CKPT" --out "$STAGE" --decode "$DECODE"
cd /home/ben/task
python evaluate.py --model-path "$STAGE" --limit "$LIMIT" --max-connections 16 \
  --gpu-memory-utilization 0.85 \
  --json-output-file "/home/ben/task/eval/${TAG}.json" > "/home/ben/task/logs/eval_${TAG}.log" 2>&1
echo "--- $TAG ---"
cat "/home/ben/task/eval/${TAG}.json"
LOG=$(ls -t /home/ben/task/logs/*.json | head -1)
cp "$LOG" "/home/ben/task/eval/${TAG}_inspect.json"
python /home/ben/task/scripts/analyze_log.py "/home/ben/task/eval/${TAG}_inspect.json" \
  --dump-failures "/home/ben/task/analysis/${TAG}_failures.jsonl"
rm -rf "$STAGE"
