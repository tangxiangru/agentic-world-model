#!/usr/bin/env bash
# exp-03: same weights as exp-02, greedy decode config. Repackage each epoch
# checkpoint with temperature 0 in generation_config.json, then read dev-150.
set -euo pipefail
cd /home/ben/task
for EP in ep1 ep2; do
  CUDA_VISIBLE_DEVICES="" python package_ckpt.py \
    --ckpt "ckpts/exp-02/${EP}" --out "ckpts/exp-03/${EP}_greedy" --decode greedy
  python evaluate.py --model-path "/home/ben/task/ckpts/exp-03/${EP}_greedy" \
    --limit 150 --max-connections 16 --gpu-memory-utilization 0.85 \
    --json-output-file "/home/ben/task/eval/exp03_${EP}_greedy_dev150.json"
  python analyze_eval.py --out "analysis/exp-03_${EP}_greedy_diag.json" \
    --watch analysis/exp-01_watch.jsonl
done
