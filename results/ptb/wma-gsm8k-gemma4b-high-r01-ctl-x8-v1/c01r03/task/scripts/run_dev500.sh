#!/bin/bash
# Greedy accuracy on our own 500-item gsm8k TRAIN holdout, scored with the
# grader's own match_str. n=500 is what separates candidates that --limit 150
# (stderr ~0.037) cannot. Never touches the benchmark test set.
cd /home/ben/task || exit 1
for ck in "$@"; do
  tag=$(echo "$ck" | tr '/' '_')
  echo "=== $ck ==="
  python scripts/sample_model.py \
    --model "$ck" \
    --questions data/dev_gsm8k_trainholdout.jsonl \
    --out "analysis/dev500_${tag}.jsonl" \
    --mode dev --max-tokens 640 --fewshot 1 \
    --gpu-mem 0.85 --max-model-len 3072 --max-num-seqs 512 2>&1 | tail -2
done
echo "DEV500_DONE"
