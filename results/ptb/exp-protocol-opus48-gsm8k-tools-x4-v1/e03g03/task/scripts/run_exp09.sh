#!/usr/bin/env bash
# exp-09 driver: STaR round-3 sampling of correct solutions from exp-08.
set -euo pipefail
cd /home/ben/task
export HF_HOME=/home/ben/hf_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
echo "=== [exp-09] sampling start $(date) ==="
python scripts/rft_sample.py --model /home/ben/task/ckpts/exp-08/final \
  --data /home/ben/task/data/gsm8k_train.jsonl \
  --raw_out /home/ben/task/data/rft3_raw.jsonl --out /home/ben/task/data/rft3_correct.jsonl \
  --n 6 --temp 1.0 --max_tokens 400 --cap 3
echo "=== [exp-09] sampling done $(date) ==="
wc -l /home/ben/task/data/rft3_correct.jsonl
