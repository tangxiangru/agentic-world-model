#!/bin/bash
cd /home/ben/task
for ck in checkpoint-234 checkpoint-702; do
  out="eval/exp01_${ck}_greedy_dev150.json"
  python evaluate.py --model-path "ckpts/exp-01/$ck" --limit 150 --max-connections 8 --json-output-file "$out" > "logs/eval_${ck}_greedy.log" 2>&1
  echo "$ck greedy: $(cat $out 2>/dev/null | tr -d '\n')"
done
echo "GREEDY_EPOCHS_DONE"
