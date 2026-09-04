#!/bin/bash
# Full-test (n=1319) greedy head-to-head: incumbent vs top exp-03 candidates.
cd /home/ben/task
declare -A M=(
  [incumbent468]="final_model"
  [exp03_240]="ckpts/exp-03/checkpoint-240"
  [exp03_840]="ckpts/exp-03/checkpoint-840"
  [exp03_858]="ckpts/exp-03/checkpoint-858"
)
for name in incumbent468 exp03_240 exp03_840 exp03_858; do
  d="${M[$name]}"
  out="eval/confirm_${name}_greedy_full1319.json"
  echo "=== evaluating $name ($d) -> $out ==="
  python evaluate.py --model-path "$d" --limit 1319 --max-connections 8 \
    --json-output-file "$out" > "logs/eval_confirm_${name}.log" 2>&1
  echo "$name exit=$? result=$(cat $out 2>/dev/null | tr -d '\n')"
done
echo "EVAL_CONFIRM_DONE"
