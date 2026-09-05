#!/usr/bin/env bash
# usage: run_exp05.sh <parent_checkpoint_dir>
set -x
set -e
cd /home/ben/task
PARENT="$1"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 1. on-policy sampling: 4 draws per question at the model's own temperature
python scripts/gen_vllm.py --model "$PARENT" --questions data/rft_questions.jsonl \
  --out analysis/exp05_samples.jsonl --n 4 --temperature 1.0 --top-p 0.95 --top-k 64 \
  --fewshot --max-tokens 768 --gpu-mem 0.85

# 2. keep the correct, well-formed ones
python scripts/build_rft_data.py --samples analysis/exp05_samples.jsonl \
  --out data/rft_v1.jsonl --per-problem 2

# 3. mix with a slice of sft_v2 so the update does not narrow the model onto its own output
python scripts/mix_rft.py --rft data/rft_v1.jsonl --sft data/sft_v2.jsonl \
  --n-sft 25000 --out data/exp05_mix.jsonl

# 4. contamination gate before anything trains on it
python scripts/decon.py --input data/exp05_mix.jsonl --tag exp05

# 5. short low-LR continuation from the parent
python scripts/train_sft.py --model "$PARENT" --data data/exp05_mix.jsonl \
  --out ckpts/exp-05 --epochs 1 --lr 5e-6 --grad-accum 4 --max-seq-len 2432 \
  --fewshot-frac 0.06 --seed 0
python scripts/finalize_model.py --src ckpts/exp-05/final --dst ckpts/exp-05/hf --greedy
bash scripts/run_eval.sh /home/ben/task/ckpts/exp-05/hf exp05_dev150 150 16
