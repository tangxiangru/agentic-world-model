#!/usr/bin/env bash
# exp-02 driver: full SFT on GSM8K-train, then eval at n=150 (same protocol as exp-01).
set -euo pipefail
cd /home/ben/task
export HF_HOME=/home/ben/hf_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

OUT=/home/ben/task/ckpts/exp-02
FINAL=$OUT/final

echo "=== [exp-02] training start $(date) ==="
python scripts/train_sft.py \
  --data /home/ben/task/data/gsm8k_train.jsonl \
  --out "$OUT" \
  --lr 1e-5 --epochs 3 --bs 4 --grad_accum 4 --max_len 1024 --seed 0
echo "=== [exp-02] training done $(date) ==="

test -f "$FINAL/config.json" || { echo "FINAL missing config"; exit 3; }
ls -la "$FINAL"

echo "=== [exp-02] eval start $(date) ==="
python evaluate.py --model-path "$FINAL" --limit 150 \
  --json-output-file /home/ben/task/eval/exp-02_n150.json
echo "=== [exp-02] eval done $(date) ==="
cat /home/ben/task/eval/exp-02_n150.json
