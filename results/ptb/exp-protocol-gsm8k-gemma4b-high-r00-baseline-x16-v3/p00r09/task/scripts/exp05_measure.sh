#!/usr/bin/env bash
# exp-05: full-test ranking of the two leading checkpoints + few-shot context probe.
set -uo pipefail
cd /home/ben/task
python scripts/probe_fewshot.py --model ckpts/exp-02/final --n 400 \
    --out analysis/exp-05_probe_exp02.json
for c in exp-02 exp-04; do
  python evaluate.py --model-path "ckpts/${c}/final" --limit 1319 \
    --max-connections 32 --max-tokens 1024 --gpu-memory-utilization 0.6 \
    --json-output-file "eval/${c}_full1319.json" > "logs/eval_${c}_full1319.log" 2>&1
  echo "== ${c} full"; cat "eval/${c}_full1319.json"
done
