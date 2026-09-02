#!/usr/bin/env bash
# exp-07: third SFT stage from exp-05/final on 95000 further fresh
# OpenMathInstruct-2 solutions, scored at --limit 500 against exp-05/final.
set -euo pipefail
cd /home/ben/task
mkdir -p eval logs ckpts

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python scripts/train_sft.py \
  --data data/sft_v4.jsonl \
  --init-from ckpts/exp-05/final \
  --output-dir ckpts/exp-07 \
  --max-seq-len 3584 \
  --lr 5e-6 \
  --epochs 1 \
  --bs 24 \
  --grad-accum 6 \
  --warmup 0.03 \
  --seed 0 \
  --save-steps 330

python scripts/fix_ckpt.py ckpts/exp-07/checkpoint-330 ckpts/exp-07/checkpoint-660 || true

python evaluate.py \
  --model-path ckpts/exp-07/final \
  --limit 500 \
  --max-connections 16 \
  --gpu-memory-utilization 0.85 \
  --json-output-file eval/exp-07_dev500.json

cat eval/exp-07_dev500.json
