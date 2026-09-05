#!/usr/bin/env bash
# exp-10: last two zero-training candidates, scored on the full 1319-item test
# set against the incumbent exp-07/final (0.71418). No training.
set -euo pipefail
cd /home/ben/task
mkdir -p eval logs

run () {
  local path="$1" out="$2"
  echo "=== ${path} ==="
  python evaluate.py \
    --model-path "${path}" \
    --limit -1 \
    --max-connections 16 \
    --gpu-memory-utilization 0.85 \
    --json-output-file "${out}"
  cat "${out}"
}

run ckpts/soup_79            eval/soup79_full.json
run ckpts/exp-07/checkpoint-330 eval/exp-07_ckpt330_full.json

echo "=== summary (incumbent exp-07/final = 0.71418) ==="
for f in eval/soup79_full.json eval/exp-07_ckpt330_full.json; do
  echo -n "$f "; tr -d '\n' < "$f"; echo
done
