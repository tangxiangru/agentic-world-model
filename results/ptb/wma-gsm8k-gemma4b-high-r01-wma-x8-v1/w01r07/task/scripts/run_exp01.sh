#!/usr/bin/env bash
# exp-01: baseline GSM8K for the immutable gemma-3-4b-pt snapshot under the
# harness protocol (--limit 150), stock sampling config vs greedy decode.
set -euo pipefail
cd /home/ben/task

SNAP=/home/ben/hf_cache/hub/models--google--gemma-3-4b-pt/snapshots/cc012e0a6d0787b4adcc0fa2c4da74402494554d
mkdir -p eval logs

echo "=== stock generation_config (T=1.0, top_p=0.95, top_k=64) ==="
python evaluate.py \
  --model-path "$SNAP" \
  --limit 150 \
  --max-connections 16 \
  --gpu-memory-utilization 0.85 \
  --json-output-file eval/base_stock_dev150.json

echo "=== greedy generation_config (T=0) ==="
python evaluate.py \
  --model-path ckpts/base_greedy \
  --limit 150 \
  --max-connections 16 \
  --gpu-memory-utilization 0.85 \
  --json-output-file eval/base_greedy_dev150.json

echo "=== done ==="
cat eval/base_stock_dev150.json
cat eval/base_greedy_dev150.json
