#!/bin/bash
# Evaluate the three epoch checkpoints at limit 150, sequentially.
cd /home/ben/task
for ck in checkpoint-234 checkpoint-468 checkpoint-702; do
  out="eval/exp01_${ck}_dev150.json"
  echo "=== evaluating $ck -> $out ==="
  python evaluate.py --model-path "ckpts/exp-01/$ck" --limit 150 \
    --max-connections 8 --json-output-file "$out" > "logs/eval_${ck}.log" 2>&1
  echo "$ck exit=$? result=$(cat $out 2>/dev/null)"
done
echo "ALL_EVALS_DONE"
