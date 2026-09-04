#!/bin/bash
export HF_HOME=/home/ben/hf_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /home/ben/task
exec python train_sft.py --out lora_v1 --n 45000 --epochs 2 --bs 8 --ga 4 --lr 2e-4 --rank 64 --maxlen 1024 --attn flash_attention_2 --gc 1
