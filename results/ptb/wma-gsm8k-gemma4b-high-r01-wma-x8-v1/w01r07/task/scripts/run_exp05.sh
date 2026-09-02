#!/usr/bin/env bash
# exp-05: second SFT stage from exp-04/final on 95000 fresh OpenMathInstruct-2
# solutions to problems already covered, then the 150-item protocol eval.
set -euo pipefail
cd /home/ben/task
mkdir -p eval logs ckpts

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python scripts/train_sft.py \
  --data data/sft_v3.jsonl \
  --init-from ckpts/exp-04/final \
  --output-dir ckpts/exp-05 \
  --max-seq-len 3584 \
  --lr 5e-6 \
  --epochs 1 \
  --bs 24 \
  --grad-accum 6 \
  --warmup 0.03 \
  --seed 0 \
  --save-steps 320

python scripts/fix_ckpt.py ckpts/exp-05/checkpoint-320 ckpts/exp-05/checkpoint-640 || true

python evaluate.py \
  --model-path ckpts/exp-05/final \
  --limit 150 \
  --max-connections 16 \
  --gpu-memory-utilization 0.85 \
  --json-output-file eval/exp-05_dev150.json

cat eval/exp-05_dev150.json
