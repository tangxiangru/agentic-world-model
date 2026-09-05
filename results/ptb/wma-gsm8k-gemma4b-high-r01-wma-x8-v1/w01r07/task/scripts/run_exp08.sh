#!/usr/bin/env bash
# exp-08: final submission choice, scored on the full 1319-item GSM8K test set.
# Arms: exp-07/final and two trajectory soups. No training.
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

run ckpts/exp-07/final eval/exp-07_full.json
run ckpts/soup_57      eval/soup57_full.json
run ckpts/soup_457     eval/soup457_full.json

echo "=== summary ==="
for f in eval/exp-07_full.json eval/soup57_full.json eval/soup457_full.json; do
  echo -n "$f "; tr -d '\n' < "$f"; echo
done
