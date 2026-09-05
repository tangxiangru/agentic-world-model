#!/usr/bin/env bash
# exp-03 driver: full SFT on GSM8K-train + MetaMathQA-GSM subset, then eval n=150.
set -euo pipefail
cd /home/ben/task
export HF_HOME=/home/ben/hf_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

OUT=/home/ben/task/ckpts/exp-03
FINAL=$OUT/final

echo "=== [exp-03] training start $(date) ==="
python scripts/train_sft.py \
  --data /home/ben/task/data/exp03_train.jsonl \
  --out "$OUT" \
  --lr 1e-5 --epochs 2 --bs 4 --grad_accum 4 --max_len 1024 --seed 0
echo "=== [exp-03] training done $(date) ==="

test -f "$FINAL/config.json" || { echo "FINAL missing config"; exit 3; }
ls -la "$FINAL"

echo "=== [exp-03] eval start $(date) ==="
python evaluate.py --model-path "$FINAL" --limit 150 \
  --json-output-file /home/ben/task/eval/exp-03_n150.json
echo "=== [exp-03] eval done $(date) ==="
cat /home/ben/task/eval/exp-03_n150.json
