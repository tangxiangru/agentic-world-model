#!/usr/bin/env bash
# exp-09: fourth SFT stage from exp-07/final on 95000 further fresh
# OpenMathInstruct-2 solutions, scored on the full 1319-item test set.
# No intermediate checkpoints: the clock, not disk, is the constraint.
set -euo pipefail
cd /home/ben/task
mkdir -p eval logs ckpts

PYTHONPATH=/home/ben/task/pylibs \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python scripts/train_sft.py \
  --data data/sft_v5.jsonl \
  --init-from ckpts/exp-07/final \
  --output-dir ckpts/exp-09 \
  --max-seq-len 3584 \
  --lr 4e-6 \
  --epochs 1 \
  --bs 24 \
  --grad-accum 6 \
  --warmup 0.03 \
  --seed 0 \
  --save-steps 0

python evaluate.py \
  --model-path ckpts/exp-09/final \
  --limit -1 \
  --max-connections 16 \
  --gpu-memory-utilization 0.85 \
  --json-output-file eval/exp-09_full.json

cat eval/exp-09_full.json
