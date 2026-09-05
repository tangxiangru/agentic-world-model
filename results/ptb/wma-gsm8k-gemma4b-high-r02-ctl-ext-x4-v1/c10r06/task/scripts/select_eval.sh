#!/usr/bin/env bash
# exp-07: score the three surviving candidates on the same 500 items, one protocol, one machine.
set -u
cd /home/ben/task
for arm in cand-a-exp02 cand-b-exp05 exp-06-soup; do
  echo "=== $arm ==="
  python evaluate.py --model-path "ckpts/$arm" --limit 500 \
    --json-output-file "/home/ben/task/eval/exp-07_${arm}_dev500.json" \
    --max-connections 16
  echo "--- $arm done: $(cat /home/ben/task/eval/exp-07_${arm}_dev500.json) ---"
done
