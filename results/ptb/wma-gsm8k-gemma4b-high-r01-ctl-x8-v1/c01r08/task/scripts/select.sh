#!/bin/bash
# exp-06: score every servable candidate under one protocol (n=500, greedy).
# Arms are run sequentially; each gets its own inspect log and metrics file.
set -u
cd /home/ben/task

ARMS=(
  "exp02:ckpts/exp-04-greedy"
  "exp05:ckpts/exp-05-final-greedy"
  "soup235:ckpts/soup-235"
  "exp03:ckpts/exp-03-greedy"
)

for arm in "${ARMS[@]}"; do
  name="${arm%%:*}"
  path="${arm#*:}"
  echo "=== $name  $path  $(date -u)"
  python evaluate.py --model-path "$path" --limit 500 --max-connections 16 \
      --json-output-file "/home/ben/task/eval/exp-06_${name}_dev500.json" \
      > "logs/exp-06_${name}.log" 2>&1
  echo "--- $name exit=$? : $(cat "/home/ben/task/eval/exp-06_${name}_dev500.json" 2>/dev/null | tr -d '\n')"
done
echo "=== done $(date -u)"
