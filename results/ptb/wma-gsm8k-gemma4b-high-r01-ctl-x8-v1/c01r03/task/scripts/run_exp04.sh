#!/bin/bash
cd /home/ben/task || exit 1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python scripts/train_sft.py \
  --model /home/ben/hf_cache/hub/models--google--gemma-3-4b-pt/snapshots/cc012e0a6d0787b4adcc0fa2c4da74402494554d \
  --data /home/ben/task/data/sft_v2.jsonl \
  --out /home/ben/task/ckpts/exp-04 \
  --epochs 1 --lr 1e-5 --bs 8 --grad-accum 4 --max-seq-len 2816 \
  --attn flash_attention_2 --save-steps 1200
echo "EXP04_TRAIN_DONE rc=$?"
