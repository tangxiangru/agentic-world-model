#!/usr/bin/env bash
# run_eval.sh <model-path> <tag> [limit] [extra evaluate.py args...]
# Runs the pinned dev protocol and writes:
#   eval/<tag>.json            metrics
#   logs/inspect_<tag>/        inspect json log
#   analysis/<tag>_items.jsonl per-item records + diagnostics
set -euo pipefail
MODEL="$1"; TAG="$2"; LIMIT="${3:-150}"; shift 3 || shift 2 || true
export HF_HOME=/home/ben/hf_cache
export INSPECT_LOG_DIR="/home/ben/task/logs/inspect_${TAG}"
rm -rf "$INSPECT_LOG_DIR"
python /home/ben/task/evaluate.py \
  --model-path "$MODEL" --limit "$LIMIT" \
  --max-connections 16 --max-tokens 1024 --gpu-memory-utilization 0.85 \
  --json-output-file "/home/ben/task/eval/${TAG}.json" "$@" \
  > "/home/ben/task/logs/eval_${TAG}.log" 2>&1
echo "--- ${TAG} ---"
cat "/home/ben/task/eval/${TAG}.json"; echo
python /home/ben/task/analyze_log.py --log-dir "$INSPECT_LOG_DIR" \
  --out "/home/ben/task/analysis/${TAG}_items.jsonl"
