#!/bin/bash
# Protocol eval, exactly as declared in every card's evaluation.protocol:
#   python evaluate.py --model-path {ckpt} --limit 150 --max-connections 16 --gpu-memory-utilization 0.75
# usage: scripts/run_protocol_eval.sh <ckpt_dir> <tag> [limit]
set -euo pipefail
CKPT="$1"; TAG="$2"; LIMIT="${3:-150}"
cd /home/ben/task
python scripts/finalize_ckpt.py --ckpt "$CKPT" >> "logs/${TAG}_eval.log" 2>&1
python evaluate.py --model-path "$CKPT" --limit "$LIMIT" \
  --max-connections 16 --gpu-memory-utilization 0.75 \
  --json-output-file "eval/${TAG}_dev${LIMIT}.json" >> "logs/${TAG}_eval.log" 2>&1
echo "--- ${TAG} ---"
cat "eval/${TAG}_dev${LIMIT}.json"
