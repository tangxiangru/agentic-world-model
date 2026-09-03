#!/bin/bash
set -x
cd /home/ben/task
while pgrep -f "train_sft.py --data data/sft_v2_mix.jsonl" > /dev/null; do sleep 20; done
sleep 10
[ -d ckpts/exp-05/final ] || { echo "NO FINAL CHECKPOINT"; exit 1; }
mkdir -p eval/logs/exp-05
INSPECT_LOG_DIR=/home/ben/task/eval/logs/exp-05 python evaluate.py \
  --model-path /home/ben/task/ckpts/exp-05/final --limit 150 \
  --max-connections 16 --gpu-memory-utilization 0.85 \
  --json-output-file /home/ben/task/eval/exp-05_dev150.json > logs/exp-05_eval.log 2>&1
python quick_eval.py --model /home/ben/task/ckpts/exp-05/final --limit 600 --fewshot 10 \
  --out /home/ben/task/eval/exp-05_privdev600_10shot.json > logs/exp-05_qeval10.log 2>&1
echo CHAIN_DONE
