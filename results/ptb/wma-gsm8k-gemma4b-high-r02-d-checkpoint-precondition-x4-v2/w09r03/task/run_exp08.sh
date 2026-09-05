#!/usr/bin/env bash
# exp-08: both surviving candidates over the same 500-item slice, one protocol.
set -u
cd /home/ben/task
python evaluate.py --model-path final_model        --limit 500 --max-connections 16 \
  --gpu-memory-utilization 0.85 --json-output-file /home/ben/task/eval/final_model_dev500.json
python evaluate.py --model-path ckpts/exp-05-greedy --limit 500 --max-connections 16 \
  --gpu-memory-utilization 0.85 --json-output-file /home/ben/task/eval/exp05_greedy_dev500.json
