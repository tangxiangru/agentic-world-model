#!/usr/bin/env bash
# exp-03: checkpoint sweep over exp-02's saved steps under the identical
# 150-item protocol. No training.
set -euo pipefail
cd /home/ben/task
mkdir -p eval logs

for step in 150 300; do
  echo "=== checkpoint-${step} ==="
  python evaluate.py \
    --model-path "ckpts/exp-02/checkpoint-${step}" \
    --limit 150 \
    --max-connections 16 \
    --gpu-memory-utilization 0.85 \
    --json-output-file "eval/exp-02_ckpt${step}_dev150.json"
done

echo "=== done ==="
cat eval/exp-02_ckpt150_dev150.json eval/exp-02_ckpt300_dev150.json eval/exp-02_dev150.json
