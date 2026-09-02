#!/usr/bin/env bash
# exp-02: full-parameter SFT of the immutable gemma-3-4b-pt snapshot on
# data/sft_v1.jsonl, then the 150-item protocol eval.
set -euo pipefail
cd /home/ben/task
mkdir -p eval logs ckpts

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python scripts/train_sft.py \
  --data data/sft_v1.jsonl \
  --output-dir ckpts/exp-02 \
  --max-seq-len 3584 \
  --lr 1e-5 \
  --epochs 1 \
  --bs 24 \
  --grad-accum 6 \
  --warmup 0.03 \
  --seed 0 \
  --save-steps 150

python evaluate.py \
  --model-path ckpts/exp-02/final \
  --limit 150 \
  --max-connections 16 \
  --gpu-memory-utilization 0.85 \
  --json-output-file eval/exp-02_dev150.json

cat eval/exp-02_dev150.json
