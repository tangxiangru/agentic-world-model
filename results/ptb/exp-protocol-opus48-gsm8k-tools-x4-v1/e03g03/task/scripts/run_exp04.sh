#!/usr/bin/env bash
# exp-04 driver: RFT/STaR sampling of correct solutions from exp-02.
set -euo pipefail
cd /home/ben/task
export HF_HOME=/home/ben/hf_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

echo "=== [exp-04] sampling start $(date) ==="
python scripts/rft_sample.py \
  --model /home/ben/task/ckpts/exp-02/final \
  --data /home/ben/task/data/gsm8k_train.jsonl \
  --raw_out /home/ben/task/data/rft_raw.jsonl \
  --out /home/ben/task/data/rft_correct.jsonl \
  --n 6 --temp 1.0 --max_tokens 400 --cap 4
echo "=== [exp-04] sampling done $(date) ==="
wc -l /home/ben/task/data/rft_correct.jsonl
