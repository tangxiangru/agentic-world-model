#!/usr/bin/env bash
# exp-06: score every finalist on the full gsm8k test split under one protocol.
set -u
cd /home/ben/task
for name in exp-02-greedy exp-04-greedy exp-05-soup; do
  echo "=== $name ==="
  date
  python evaluate.py \
    --model-path "/home/ben/task/ckpts/${name}" \
    --limit 1319 \
    --max-connections 16 \
    --gpu-memory-utilization 0.85 \
    --json-output-file "/home/ben/task/eval/exp-06_${name}_full1319.json"
  echo "--- $name result ---"
  cat "/home/ben/task/eval/exp-06_${name}_full1319.json"
done
date
echo "SELECTION DONE"
