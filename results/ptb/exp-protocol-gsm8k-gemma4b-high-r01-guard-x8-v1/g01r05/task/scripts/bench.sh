#!/bin/bash
# quick throughput probe: prints s/it for a few configs
set -x
M=/home/ben/hf_cache/hub/models--google--gemma-3-4b-pt/snapshots/cc012e0a6d0787b4adcc0fa2c4da74402494554d
for cfg in "float32 16 1 gc" "bfloat16 32 1 gc" "bfloat16 16 1 nogc"; do
  set -- $cfg
  python scripts/train_sft.py --model $M --data data/sft_v1.jsonl --out /tmp/bench_$1_$2_$4 \
    --limit 4000 --max-steps 12 --bs $2 --grad-accum $3 --model-dtype $1 \
    ${4:+$([ "$4" = nogc ] && echo --no-grad-ckpt)} 2>&1 | grep -E "train_runtime|train_samples_per_second|MemErr|CUDA out of memory|\[model\]" | tail -5
  echo "=== done $cfg ==="
done
