#!/usr/bin/env bash
# exp-07: read the exp-04/exp-05 weight average under the exp-06 protocol.
set -euo pipefail
cd /home/ben/task
python evaluate.py --model-path /home/ben/task/ckpts/exp-07/soup \
  --limit 500 --max-connections 16 --gpu-memory-utilization 0.85 \
  --json-output-file /home/ben/task/eval/exp07_soup_dev500.json
python analyze_eval.py --out analysis/exp-07_soup_dev500_diag.json
