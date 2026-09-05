#!/bin/bash
set -x
cd /home/ben/task
while pgrep -f "train_sft.py --data data/sft_v3_mix.jsonl" > /dev/null; do sleep 20; done
sleep 10
[ -d ckpts/exp-06/final ] || { echo "NO FINAL CHECKPOINT"; exit 1; }
mkdir -p eval/logs/exp-06
INSPECT_LOG_DIR=/home/ben/task/eval/logs/exp-06 python evaluate.py \
  --model-path /home/ben/task/ckpts/exp-06/final --limit 150 \
  --max-connections 16 --gpu-memory-utilization 0.85 \
  --json-output-file /home/ben/task/eval/exp-06_dev150.json > logs/exp-06_eval.log 2>&1
python quick_eval.py --model /home/ben/task/ckpts/exp-06/final --limit 600 --fewshot 10 \
  --out /home/ben/task/eval/exp-06_privdev600_10shot.json > logs/exp-06_qeval10.log 2>&1
echo CHAIN_DONE
