#!/usr/bin/env bash
# exp-06: read both surviving candidates at n=500 under one protocol and ship the winner.
set -euo pipefail
cd /home/ben/task
for M in exp-04 exp-05; do
  python evaluate.py --model-path "/home/ben/task/ckpts/${M}/greedy" \
    --limit 500 --max-connections 16 --gpu-memory-utilization 0.85 \
    --json-output-file "/home/ben/task/eval/${M}_greedy_dev500.json"
  python analyze_eval.py --out "analysis/exp-06_${M}_dev500_diag.json"
done
