#!/usr/bin/env bash
# Score the surviving candidates on the first 500 test items, greedy, one
# vLLM process each. exp-04/best is skipped: exp-05 already read it under this
# exact protocol and that read is this card's comparator.
set -u
cd "$(dirname "$0")/.."
for d in ckpts/exp-04/best381 ckpts/exp-04/best350 ckpts/exp-03/best; do
  tag=$(echo "$d" | sed 's#ckpts/##; s#/#_#g')
  out="eval/${tag}_dev500.json"
  [ -s "$out" ] && { echo "$tag (cached): $(cat "$out")"; continue; }
  PYTHONDONTWRITEBYTECODE=1 INSPECT_LOG_DIR="/home/ben/task/eval/logs/${tag}_500" \
    python evaluate.py --model-path "/home/ben/task/$d" --limit 500 --max-connections 2 \
    --json-output-file "/home/ben/task/$out" > "logs/eval_${tag}_500.log" 2>&1
  echo "$tag: $(cat "$out" 2>/dev/null || echo FAILED)"
done
echo "--- comparator (exp-05) ---"; cat eval/final_dev500.json
