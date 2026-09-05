#!/bin/bash
# exp-07: full test-set reads (--limit -1, 1319 items) of the two candidates tied at 0.6933 on n=150.
set -e
cd /home/ben/task
for NAME in exp-02-ck1800-greedy exp-04-final-greedy; do
  mkdir -p eval/inspect_logs/exp-07-$NAME
  INSPECT_LOG_DIR=/home/ben/task/eval/inspect_logs/exp-07-$NAME \
  python evaluate.py \
    --model-path /home/ben/task/ckpts/$NAME \
    --limit -1 \
    --json-output-file /home/ben/task/eval/exp-07_${NAME}_full.json \
    --max-connections 2 \
    --gpu-memory-utilization 0.85
  python work/analyse_eval.py eval/inspect_logs/exp-07-$NAME --out analysis/exp-07_$NAME.json
done
