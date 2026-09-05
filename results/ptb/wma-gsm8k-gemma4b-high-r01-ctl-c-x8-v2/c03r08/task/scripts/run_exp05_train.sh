#!/usr/bin/env bash
set -x
set -e
cd /home/ben/task
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python scripts/train_sft.py --model /home/ben/task/ckpts/exp-04/hf --data data/exp05_mix.jsonl \
  --out ckpts/exp-05 --epochs 1 --lr 5e-6 --grad-accum 4 --max-seq-len 2432 \
  --fewshot-frac 0.06 --seed 0
python scripts/finalize_model.py --src ckpts/exp-05/final --dst ckpts/exp-05/hf --greedy
bash scripts/run_eval.sh /home/ben/task/ckpts/exp-05/hf exp05_dev150 150 16
