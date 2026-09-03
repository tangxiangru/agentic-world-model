#!/bin/bash
# Chain exp-02's evaluation straight onto the end of its training run so the
# H100 never sits idle waiting for me to notice the trainer exited.
set -x
cd /home/ben/task
while pgrep -f "train_sft.py --data data/sft_v1.jsonl" > /dev/null; do sleep 20; done
sleep 10
if [ ! -d ckpts/exp-02/final ]; then echo "NO FINAL CHECKPOINT"; exit 1; fi
mkdir -p eval/logs/exp-02
INSPECT_LOG_DIR=/home/ben/task/eval/logs/exp-02 python evaluate.py \
  --model-path /home/ben/task/ckpts/exp-02/final --limit 150 \
  --max-connections 16 --gpu-memory-utilization 0.85 \
  --json-output-file /home/ben/task/eval/exp-02_dev150.json > logs/exp-02_eval.log 2>&1
python quick_eval.py --model /home/ben/task/ckpts/exp-02/final --limit 600 --fewshot 10 \
  --out /home/ben/task/eval/exp-02_privdev600_10shot.json > logs/exp-02_qeval10.log 2>&1
python quick_eval.py --model /home/ben/task/ckpts/exp-02/final --limit 600 --fewshot 0 \
  --out /home/ben/task/eval/exp-02_privdev600_0shot.json > logs/exp-02_qeval0.log 2>&1
echo CHAIN_DONE
