#!/usr/bin/env bash
# exp-07: two extra n=1319 reads of each finalist under the frozen argv,
# interleaved so any machine-state drift hits both candidates equally.
set -u
cd /home/ben/task

for r in 1 2; do
  for c in exp-04 exp-06; do
    tag="${c}_rep${r}"
    echo "=== $tag ==="
    INSPECT_LOG_DIR="logs/rep/$tag" timeout 3000 python evaluate.py \
      --model-path "ckpts/$c/final" \
      --limit 1319 --max-connections 8 --gpu-memory-utilization 0.85 \
      --json-output-file "/home/ben/task/eval/${tag}.json" \
      > "logs/rep_${tag}.log" 2>&1
    echo "exit $?  $(grep -oE 'accuracy +[0-9.]+' "logs/rep_${tag}.log" | tail -1)"
    python scripts/dump_eval.py --log logs/rep/$tag/*.json \
      --out "analysis/${tag}_items.jsonl" 2>&1 | grep -E "^accuracy|^items"
  done
done
echo "=== all reads done ==="
