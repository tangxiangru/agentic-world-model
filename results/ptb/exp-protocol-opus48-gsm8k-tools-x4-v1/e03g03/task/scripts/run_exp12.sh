#!/usr/bin/env bash
# exp-12 driver: 1-epoch SFT from base on the 3-round RFT data (salvage exp-10's reasoning, preserve stopping).
set -euo pipefail
cd /home/ben/task
export HF_HOME=/home/ben/hf_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
OUT=/home/ben/task/ckpts/exp-12; FINAL=$OUT/final
echo "=== [exp-12] training start $(date) ==="
python scripts/train_sft.py --data /home/ben/task/data/exp10_train.jsonl --out "$OUT" \
  --lr 1e-5 --epochs 1 --bs 4 --grad_accum 4 --max_len 1024 --seed 0
echo "=== [exp-12] training done $(date) ==="
test -f "$FINAL/config.json" || { echo "FINAL missing config"; exit 3; }
echo "=== [exp-12] eval start $(date) ==="
python evaluate.py --model-path "$FINAL" --limit 150 --json-output-file /home/ben/task/eval/exp-12_n150.json
echo "=== [exp-12] eval done $(date) ==="
cat /home/ben/task/eval/exp-12_n150.json
