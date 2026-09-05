#!/usr/bin/env bash
# exp-06: candidate selection at --limit 500 (SE ~0.020 instead of 0.037).
# Arms: exp-04/final, exp-05/final, and their uniform weight soup.
set -euo pipefail
cd /home/ben/task
mkdir -p eval logs

run () {
  local path="$1" out="$2"
  echo "=== ${path} ==="
  python evaluate.py \
    --model-path "${path}" \
    --limit 500 \
    --max-connections 16 \
    --gpu-memory-utilization 0.85 \
    --json-output-file "${out}"
  cat "${out}"
}

run ckpts/exp-05/final  eval/exp-05_dev500.json
run ckpts/exp-04/final  eval/exp-04_dev500.json
run ckpts/soup_45       eval/soup45_dev500.json

echo "=== summary ==="
for f in eval/exp-05_dev500.json eval/exp-04_dev500.json eval/soup45_dev500.json; do
  echo -n "$f "; cat "$f" | tr -d '\n'; echo
done
