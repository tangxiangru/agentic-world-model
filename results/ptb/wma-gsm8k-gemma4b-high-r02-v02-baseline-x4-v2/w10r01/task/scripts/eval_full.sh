#!/usr/bin/env bash
# Read the two surviving candidates on the whole gsm8k test split (--limit -1
# disables the limit in evaluate.py), because the first-150 slice and the
# first-500 slice rank them in opposite orders.
set -u
cd "$(dirname "$0")/.."
for d in ckpts/soup_final ckpts/exp-04/best; do
  tag=$(echo "$d" | sed 's#ckpts/##; s#/#_#g')
  out="eval/${tag}_full.json"
  [ -s "$out" ] && { echo "$tag (cached): $(cat "$out")"; continue; }
  PYTHONDONTWRITEBYTECODE=1 INSPECT_LOG_DIR="/home/ben/task/eval/logs/${tag}_full" \
    python evaluate.py --model-path "/home/ben/task/$d" --limit -1 --max-connections 2 \
    --json-output-file "/home/ben/task/$out" > "logs/eval_${tag}_full.log" 2>&1
  echo "$tag: $(cat "$out" 2>/dev/null || echo FAILED)"
done
