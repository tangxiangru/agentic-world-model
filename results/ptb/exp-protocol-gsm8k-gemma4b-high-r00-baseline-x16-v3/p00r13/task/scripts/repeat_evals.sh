#!/usr/bin/env bash
# exp-06: repeated full-test evals of the two leading candidates.
# vLLM greedy decoding is not run-to-run deterministic (batch-dependent kernel
# numerics), so each run is one draw; average them before choosing final_model.
set -euo pipefail
cd /home/ben/task
for i in 1 2; do
  bash scripts/run_eval.sh /home/ben/task/ckpts/exp-03-greedy       "exp-06_exp03_full_r${i}"  1319
  bash scripts/run_eval.sh /home/ben/task/ckpts/exp-04/checkpoint-550 "exp-06_ck550_full_r${i}" 1319
done
