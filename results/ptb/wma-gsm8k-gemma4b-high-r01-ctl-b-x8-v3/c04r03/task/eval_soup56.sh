#!/bin/bash
set -x
cd /home/ben/task
mkdir -p eval/logs/exp-07
INSPECT_LOG_DIR=/home/ben/task/eval/logs/exp-07 python evaluate.py \
  --model-path /home/ben/task/ckpts/soup56 --limit 150 \
  --max-connections 16 --gpu-memory-utilization 0.85 \
  --json-output-file /home/ben/task/eval/exp-07_dev150.json > logs/exp-07_eval.log 2>&1
python quick_eval.py --model /home/ben/task/ckpts/soup56 --limit 600 --fewshot 10 \
  --out /home/ben/task/eval/exp-07_privdev600_10shot.json > logs/exp-07_qeval10.log 2>&1
echo CHAIN_DONE
