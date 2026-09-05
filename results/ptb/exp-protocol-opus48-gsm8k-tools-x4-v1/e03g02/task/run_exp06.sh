set -uo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "=== STAGE1 SAMPLE $(date) ==="
timeout 2000 python sample_rft.py --model /home/ben/task/ckpts/exp-04/merged --out data/rft_correct.jsonl --n 4 --k-shot 2 --temp 0.8 --max-tokens 400 --keep-per-q 2 --seed 0
SRC=$?; echo "sample rc=$SRC"; [ $SRC -ne 0 ] && { echo "SAMPLE FAILED/TIMEOUT"; exit 31; }
echo "=== STAGE2 CONTAM $(date) ==="
if ! python ../contamination_check.py --reference ../test_data.json --input data/rft_correct.jsonl > analysis/contam_rft.txt 2>&1; then
  echo "CONTAMINATION -> ABORT"; grep "Contaminated documents" analysis/contam_rft.txt; exit 33; fi
grep -E "Contaminated documents|Documents scanned" analysis/contam_rft.txt
echo "=== STAGE3 BUILD $(date) ==="
cat data/gsm8k_train_fewshot.jsonl data/rft_correct.jsonl > data/combined_rft.jsonl
wc -l data/rft_correct.jsonl data/combined_rft.jsonl
echo "=== STAGE4 TRAIN (continue from exp-04) $(date) ==="
timeout 6000 python train_sft.py --model /home/ben/task/ckpts/exp-04/merged --data data/combined_rft.jsonl --out /home/ben/task/ckpts/exp-06/merged --epochs 1 --lr 1e-4 --bs 4 --grad-accum 8 --max-seq-len 1536 --lora-r 64 --lora-alpha 128 --seed 0
echo "train rc=$?"; echo "=== DONE $(date) ==="
