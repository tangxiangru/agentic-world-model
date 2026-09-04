#!/bin/bash
# Usage: run_eval.sh MODEL_PATH LIMIT OUTJSON
export HF_HOME=/home/ben/hf_cache
cd /home/ben/task
MODEL="$1"
LIMIT="${2:-150}"
OUT="${3:-metrics.json}"
python evaluate.py --model-path "$MODEL" --limit "$LIMIT" --max-connections 16 --gpu-memory-utilization 0.6 --json-output-file "$OUT"
