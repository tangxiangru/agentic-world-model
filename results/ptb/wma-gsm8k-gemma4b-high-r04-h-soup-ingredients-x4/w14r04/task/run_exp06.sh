#!/bin/bash
# exp-06: two pre-specified wider model soups over the same SFT trajectory.
set -euo pipefail
cd /home/ben/task

# Variant A - widen the span: two checkpoints from each of the three runs.
python merge_models.py \
  --srcs ckpts/exp-02/checkpoint-1200 ckpts/exp-02/final \
         ckpts/exp-03/checkpoint-1000 ckpts/exp-03/final \
         ckpts/exp-04/checkpoint-300  ckpts/exp-04/final \
  --dst ckpts/soupA
python finalize_model.py --src ckpts/soupA --dst ckpts/soupA_eval

# Variant B - later half only: does dropping the weakest member (exp-02) help?
python merge_models.py \
  --srcs ckpts/exp-03/checkpoint-1000 ckpts/exp-03/final \
         ckpts/exp-04/checkpoint-300  ckpts/exp-04/final \
  --dst ckpts/soupB
python finalize_model.py --src ckpts/soupB --dst ckpts/soupB_eval

python probe_eval.py --model ckpts/soupA_eval --limit 300 --out analysis/exp-06_soupA_probe.json
python probe_eval.py --model ckpts/soupB_eval --limit 300 --out analysis/exp-06_soupB_probe.json
