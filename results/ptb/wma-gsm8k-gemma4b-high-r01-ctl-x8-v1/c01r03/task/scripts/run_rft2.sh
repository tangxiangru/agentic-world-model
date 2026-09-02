#!/bin/bash
cd /home/ben/task || exit 1
python scripts/sample_model.py \
  --model ckpts/exp-05/final \
  --questions data/rft_q_gsm8k.jsonl \
  --out data/rft2_samples.jsonl \
  --mode rft --k 4 --temperature 0.7 --top-p 0.95 --top-k 64 --max-tokens 600 \
  --fewshot 0 --gpu-mem 0.85 --max-model-len 1536 --max-num-seqs 1024
echo "RFT2_DONE rc=$?"
