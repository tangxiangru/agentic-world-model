set -euo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "=== STAGE 1 SAMPLE $(date) ==="
python sample_rft.py --model /home/ben/task/ckpts/exp-04/merged --out data/rft_correct.jsonl --rec-dir sampling/rft_run --card memory/cards/exp-05.yaml --n 4 --k-shot 4 --temp 0.8 --max-tokens 400 --keep-per-q 2 --seed 0
echo "=== STAGE 2 CONTAMINATION CHECK $(date) ==="
if ! python ../contamination_check.py --reference ../test_data.json --input data/rft_correct.jsonl > analysis/contam_rft.txt 2>&1; then
  echo "CONTAMINATION DETECTED in rft_correct.jsonl -- ABORT"; grep -E "Contaminated documents" analysis/contam_rft.txt; exit 33
fi
grep -E "Contaminated documents|Documents scanned" analysis/contam_rft.txt
echo "=== STAGE 3 BUILD COMBINED $(date) ==="
cat data/gsm8k_train_fewshot.jsonl data/rft_correct.jsonl > data/combined_rft.jsonl
wc -l data/gsm8k_train_fewshot.jsonl data/rft_correct.jsonl data/combined_rft.jsonl
echo "=== STAGE 4 TRAIN $(date) ==="
python train_sft.py --model /home/ben/hf_cache/hub/models--google--gemma-3-4b-pt/snapshots/cc012e0a6d0787b4adcc0fa2c4da74402494554d --data data/combined_rft.jsonl --out /home/ben/task/ckpts/exp-05/merged --epochs 2 --lr 2e-4 --bs 4 --grad-accum 8 --max-seq-len 1536 --lora-r 64 --lora-alpha 128 --seed 0
echo "=== DONE $(date) ==="
