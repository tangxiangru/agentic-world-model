#!/usr/bin/env bash
set -x
cd /home/ben/task
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python scripts/train_sft.py \
  --model /home/ben/hf_cache/hub/models--google--gemma-3-4b-pt/snapshots/cc012e0a6d0787b4adcc0fa2c4da74402494554d \
  --data data/sft_v2.jsonl --out ckpts/exp-04 --epochs 2 --lr 1e-5 --grad-accum 4 \
  --max-seq-len 2432 --fewshot-frac 0.06 --seed 0
python scripts/finalize_model.py --src ckpts/exp-04/final --dst ckpts/exp-04/hf --greedy
bash scripts/run_eval.sh /home/ben/task/ckpts/exp-04/hf exp04_dev150 150 16
