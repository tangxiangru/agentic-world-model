#!/bin/bash
# exp-05: score both candidates on the same 500 test items, one after the other.
set -x
cd /home/ben/task
INSPECT_LOG_DIR=/home/ben/task/eval/logs_exp03_500 python evaluate.py \
  --model-path /home/ben/task/ckpts/exp-03/final --limit 500 \
  --max-connections 8 --gpu-memory-utilization 0.85 \
  --json-output-file /home/ben/task/eval/exp-03_dev500.json
INSPECT_LOG_DIR=/home/ben/task/eval/logs_exp04_500 python evaluate.py \
  --model-path /home/ben/task/ckpts/exp-04/final --limit 500 \
  --max-connections 8 --gpu-memory-utilization 0.85 \
  --json-output-file /home/ben/task/eval/exp-04_dev500.json
echo BOTH_DONE
