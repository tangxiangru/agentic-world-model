#!/usr/bin/env bash
# usage: run_eval.sh <model-path> <tag> [limit]
set -u
MP="$1"; TAG="$2"; LIM="${3:-150}"
cd /home/ben/task
BEFORE=$(ls -1 logs/*_gsm8k_*.json 2>/dev/null | wc -l)
python evaluate.py --model-path "$MP" --limit "$LIM" --max-connections 16 \
  --gpu-memory-utilization 0.85 \
  --json-output-file "/home/ben/task/eval/${TAG}.json" > "logs/${TAG}.log" 2>&1
echo "exit=$?" >> "logs/${TAG}.log"
NEW=$(ls -1t logs/*_gsm8k_*.json | head -1)
echo "inspect log: $NEW"
python scripts/analyze_log.py "$NEW" "analysis/${TAG}_diag.json"
cat "eval/${TAG}.json"
