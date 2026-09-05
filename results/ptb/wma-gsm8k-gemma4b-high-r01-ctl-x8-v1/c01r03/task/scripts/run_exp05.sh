#!/bin/bash
cd /home/ben/task || exit 1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python scripts/train_sft.py \
  --model /home/ben/task/ckpts/exp-04/final \
  --data /home/ben/task/data/sft_v3.jsonl \
  --out /home/ben/task/ckpts/exp-05 \
  --epochs 1 --lr 5e-6 --bs 8 --grad-accum 4 --max-seq-len 2816 \
  --warmup 0.02 --attn flash_attention_2 --save-steps 0
echo "EXP05_TRAIN_DONE rc=$?"
