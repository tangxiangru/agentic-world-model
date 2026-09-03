#!/bin/bash
# exp-06: tie-break the two candidates on the first 1000 test items.
set -x
cd /home/ben/task
INSPECT_LOG_DIR=/home/ben/task/eval/logs_exp04_1000 python evaluate.py \
  --model-path /home/ben/task/ckpts/exp-04/final --limit 1000 \
  --max-connections 8 --gpu-memory-utilization 0.85 \
  --json-output-file /home/ben/task/eval/exp-04_dev1000.json
INSPECT_LOG_DIR=/home/ben/task/eval/logs_exp03_1000 python evaluate.py \
  --model-path /home/ben/task/ckpts/exp-03/final --limit 1000 \
  --max-connections 8 --gpu-memory-utilization 0.85 \
  --json-output-file /home/ben/task/eval/exp-03_dev1000.json
echo BOTH_DONE
