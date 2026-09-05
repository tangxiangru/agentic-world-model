#!/usr/bin/env bash
# Final checkpoint selection: score every surviving candidate under the grading
# protocol (dev-150 greedy) and on the 250-item held-out probe.
set -x
cd /home/ben/task
python scripts/soup.py --inputs ckpts/exp-02/final ckpts/exp-04/final ckpts/exp-05/final \
  --out ckpts/soup245/raw
python scripts/finalize_model.py --src ckpts/soup245/raw --dst ckpts/soup245/hf --greedy
for M in soup24 soup245; do
  python scripts/gen_vllm.py --model ckpts/$M/hf --questions data/probe250.jsonl \
    --out analysis/exp06_probe_$M.jsonl --n 1 --temperature 0.0 --fewshot --max-tokens 1024
done
python scripts/gen_vllm.py --model ckpts/exp-05/hf --questions data/probe250.jsonl \
  --out analysis/exp06_probe_exp05.jsonl --n 1 --temperature 0.0 --fewshot --max-tokens 1024
bash scripts/run_eval.sh /home/ben/task/ckpts/soup24/hf exp06_soup24_dev150 150 16
bash scripts/run_eval.sh /home/ben/task/ckpts/soup245/hf exp06_soup245_dev150 150 16
