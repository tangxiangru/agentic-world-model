#!/bin/bash
set -x
cd /home/ben/task
mkdir -p eval/logs/exp-04
INSPECT_LOG_DIR=/home/ben/task/eval/logs/exp-04 python evaluate.py \
  --model-path /home/ben/task/ckpts/soup23 --limit 150 \
  --max-connections 16 --gpu-memory-utilization 0.85 \
  --json-output-file /home/ben/task/eval/exp-04_dev150.json > logs/exp-04_eval.log 2>&1
python quick_eval.py --model /home/ben/task/ckpts/soup23 --limit 600 --fewshot 10 \
  --out /home/ben/task/eval/exp-04_privdev600_10shot.json > logs/exp-04_qeval10.log 2>&1
echo CHAIN_DONE
