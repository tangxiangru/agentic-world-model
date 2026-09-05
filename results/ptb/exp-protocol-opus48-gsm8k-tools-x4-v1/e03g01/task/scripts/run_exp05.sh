#!/bin/bash
set -o pipefail
cd /home/ben/task
export HF_HOME=/home/ben/hf_cache
export VLLM_LOGGING_LEVEL=WARNING
export TOKENIZERS_PARALLELISM=false

echo "=== TRAIN START $(date) ==="
python scripts/train_sft.py --card memory/cards/exp-05.yaml \
  --model /home/ben/hf_cache/hub/models--google--gemma-3-4b-pt/snapshots/cc012e0a6d0787b4adcc0fa2c4da74402494554d \
  --output /home/ben/task/ckpts/exp-05 --lr 2e-5 --epochs 2 \
  --per-device-batch 16 --grad-accum 2 --attn eager
TRAIN_EXIT=$?
echo "=== TRAIN EXIT=$TRAIN_EXIT $(date) ==="
if [ $TRAIN_EXIT -ne 0 ]; then echo "RUNNER_DONE exit=$TRAIN_EXIT"; exit $TRAIN_EXIT; fi

echo "=== EVAL START $(date) ==="
python evaluate.py --model-path /home/ben/task/ckpts/exp-05 --limit 150 \
  --max-connections 32 --gpu-memory-utilization 0.85 \
  --json-output-file /home/ben/task/eval/exp-05_dev150.json
EVAL_EXIT=$?
echo "=== EVAL EXIT=$EVAL_EXIT $(date) ==="
echo "=== RESULT ==="; cat /home/ben/task/eval/exp-05_dev150.json 2>/dev/null; echo ""
echo "RUNNER_DONE exit=$EVAL_EXIT"
