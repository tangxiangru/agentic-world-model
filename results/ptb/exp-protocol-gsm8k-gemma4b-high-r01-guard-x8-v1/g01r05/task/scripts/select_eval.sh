#!/bin/bash
# exp-05: score both candidate checkpoints on the full official test set under one protocol.
set -e
cd /home/ben/task
for name in exp-02 exp-04; do
  python evaluate.py --model-path /home/ben/task/ckpts/$name/final --limit 1319 \
    --max-connections 16 --gpu-memory-utilization 0.85 \
    --json-output-file /home/ben/task/eval/${name}_n1319.json
  echo "=== $name done ==="
done
