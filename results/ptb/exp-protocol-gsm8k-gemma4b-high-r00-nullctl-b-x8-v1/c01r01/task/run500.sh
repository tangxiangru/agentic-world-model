#!/bin/bash
set -u
python evaluate.py --model-path work/export_last --limit 500 --max-connections 32 \
  --gpu-memory-utilization 0.8 --json-output-file work/grpo_last_500.json > logs/eval_last500.log 2>&1
echo "last(ckpt-300): $(cat work/grpo_last_500.json)"
python evaluate.py --model-path work/export_mid --limit 500 --max-connections 32 \
  --gpu-memory-utilization 0.8 --json-output-file work/grpo_mid_500.json > logs/eval_mid500.log 2>&1
echo "mid(ckpt-200): $(cat work/grpo_mid_500.json)"
