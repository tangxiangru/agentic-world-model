#!/usr/bin/env bash
set -x
set -e
cd /home/ben/task
PARENT="$1"
python scripts/gen_vllm.py --model "$PARENT" --questions data/rft_questions.jsonl \
  --out analysis/exp05_samples.jsonl --n 4 --temperature 1.0 --top-p 0.95 --top-k 64 \
  --fewshot --max-tokens 768 --gpu-mem 0.85
python scripts/build_rft_data.py --samples analysis/exp05_samples.jsonl \
  --out data/rft_v1.jsonl --per-problem 2
python scripts/mix_rft.py --rft data/rft_v1.jsonl --sft data/sft_v2.jsonl \
  --n-sft 25000 --out data/exp05_mix.jsonl
python scripts/decon.py --input data/exp05_mix.jsonl --tag exp05
