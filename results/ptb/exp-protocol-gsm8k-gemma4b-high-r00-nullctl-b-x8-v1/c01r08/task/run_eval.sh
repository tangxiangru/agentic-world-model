#!/bin/bash
# usage: run_eval.sh MODEL_PATH NAME [LIMIT] [MAXCONN]
M=$1; N=$2; L=${3:-150}; C=${4:-16}
python evaluate.py --model-path "$M" --limit $L --max-connections $C \
  --gpu-memory-utilization 0.55 --max-tokens 1024 \
  --json-output-file runs/$N.json > logs/eval_$N.log 2>&1
echo "== $N =="; cat runs/$N.json
