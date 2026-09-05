#!/usr/bin/env bash
# exp-06: score the three healthy candidates at n=500 under one protocol.
set -u
cd /home/ben/task
for arm in "exp02:/home/ben/task/ckpts/exp-02/final_greedy" \
           "exp04:/home/ben/task/ckpts/exp-04/final_greedy" \
           "soup:/home/ben/task/ckpts/soup_0202_04"; do
  name="${arm%%:*}"; path="${arm#*:}"
  echo "=== $name $path ==="
  python evaluate.py --model-path "$path" --limit 500 \
    --max-connections 16 --gpu-memory-utilization 0.85 \
    --json-output-file "/home/ben/task/eval/exp06_${name}_dev500.json" \
    > "/home/ben/task/logs/exp-06_${name}.log" 2>&1
  cp "$(ls -t /home/ben/task/logs/*gsm8k*.json | head -1)" \
     "/home/ben/task/eval/exp06_${name}_dev500_log.json"
  cat "/home/ben/task/eval/exp06_${name}_dev500.json"
  echo
done
