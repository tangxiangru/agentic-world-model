#!/bin/bash
# exp-05: greedy n=150 reads of exp-02's unread intermediate checkpoints.
set -e
cd /home/ben/task
for CK in ck900 ck1800; do
  mkdir -p eval/inspect_logs/exp-05-$CK
  INSPECT_LOG_DIR=/home/ben/task/eval/inspect_logs/exp-05-$CK \
  python evaluate.py \
    --model-path /home/ben/task/ckpts/exp-02-$CK-greedy \
    --limit 150 \
    --json-output-file /home/ben/task/eval/exp-05_${CK}_dev150.json \
    --max-connections 2 \
    --gpu-memory-utilization 0.85
done
