#!/usr/bin/env bash
# exp-04: same recipe as exp-02 from the immutable snapshot, on 1.8x the unique
# gsm8k-derived problems (data/sft_v2.jsonl), then the 150-item protocol eval.
set -euo pipefail
cd /home/ben/task
mkdir -p eval logs ckpts

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python scripts/train_sft.py \
  --data data/sft_v2.jsonl \
  --output-dir ckpts/exp-04 \
  --max-seq-len 3584 \
  --lr 1e-5 \
  --epochs 1 \
  --bs 24 \
  --grad-accum 6 \
  --warmup 0.03 \
  --seed 0 \
  --save-steps 220

python scripts/fix_ckpt.py ckpts/exp-04/checkpoint-220 ckpts/exp-04/checkpoint-440 || true

python evaluate.py \
  --model-path ckpts/exp-04/final \
  --limit 150 \
  --max-connections 16 \
  --gpu-memory-utilization 0.85 \
  --json-output-file eval/exp-04_dev150.json

cat eval/exp-04_dev150.json
