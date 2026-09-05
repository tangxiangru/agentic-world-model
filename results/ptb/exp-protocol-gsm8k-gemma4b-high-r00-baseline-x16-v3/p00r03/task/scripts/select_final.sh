#!/bin/bash
# exp-05: score the three candidates on 500 benchmark items under one protocol.
set -x
cd /home/ben/task
for c in exp03_final:ckpts/exp-03/final exp04_ck260:ckpts/exp-04/checkpoint-260 exp04_final:ckpts/exp-04/final; do
  name="${c%%:*}"; path="${c#*:}"
  python evaluate.py --model-path "$path" --limit 500 --max-connections 16 \
    --json-output-file "eval/${name}_official500.json"
done
echo "=== SELECT DONE ==="
