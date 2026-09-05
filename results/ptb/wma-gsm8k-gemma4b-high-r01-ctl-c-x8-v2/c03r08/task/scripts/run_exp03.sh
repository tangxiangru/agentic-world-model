#!/usr/bin/env bash
set -x
cd /home/ben/task
python scripts/make_decode_variant.py --src ckpts/exp-02/hf --dst ckpts/exp-02/hf_greedy --temperature 0.0
bash scripts/run_eval.sh /home/ben/task/ckpts/exp-02/hf_greedy exp03_dev150 150 16
python scripts/gen_vllm.py --model ckpts/exp-02/hf --questions data/probe250.jsonl \
  --out analysis/exp03_probe_greedy.jsonl --n 1 --temperature 0.0 --fewshot --max-tokens 1024
python scripts/gen_vllm.py --model ckpts/exp-02/hf --questions data/probe250.jsonl \
  --out analysis/exp03_probe_sampled.jsonl --n 1 --temperature 1.0 --top-p 0.95 --top-k 64 --fewshot --max-tokens 1024
