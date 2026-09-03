#!/usr/bin/env bash
# exp-08: full-split (1319-item) read of the shipped artifact and the runner-up.
# final_model/ is evaluated by its own path, not by the ckpts/ path, so the read
# exercises exactly what the grader will load.
set -euo pipefail
cd /home/ben/task
python evaluate.py --model-path final_model \
  --limit -1 --max-connections 16 --gpu-memory-utilization 0.85 \
  --json-output-file /home/ben/task/eval/final_model_full.json
python analyze_eval.py --out analysis/exp-08_final_model_full_diag.json
python evaluate.py --model-path /home/ben/task/ckpts/exp-07/soup \
  --limit -1 --max-connections 16 --gpu-memory-utilization 0.85 \
  --json-output-file /home/ben/task/eval/exp07_soup_full.json
python analyze_eval.py --out analysis/exp-08_soup_full_diag.json
