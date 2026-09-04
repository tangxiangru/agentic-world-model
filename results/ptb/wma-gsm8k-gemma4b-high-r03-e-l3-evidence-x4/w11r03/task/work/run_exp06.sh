#!/bin/bash
# exp-06: uniform weight average of three checkpoints on the exp-02/exp-04 tree, graded greedy.
set -e
cd /home/ben/task
python work/merge_ckpts.py \
  --src /home/ben/task/ckpts/exp-02/checkpoint-1800 \
        /home/ben/task/ckpts/exp-02/final \
        /home/ben/task/ckpts/exp-04/final \
  --dst /home/ben/task/ckpts/exp-06-soup \
  --temperature 0.0
mkdir -p eval/inspect_logs/exp-06
INSPECT_LOG_DIR=/home/ben/task/eval/inspect_logs/exp-06 \
python evaluate.py \
  --model-path /home/ben/task/ckpts/exp-06-soup \
  --limit 150 \
  --json-output-file /home/ben/task/eval/exp-06_soup_dev150.json \
  --max-connections 2 \
  --gpu-memory-utilization 0.85
python work/analyse_eval.py eval/inspect_logs/exp-06 --out analysis/exp-06_soup.json
