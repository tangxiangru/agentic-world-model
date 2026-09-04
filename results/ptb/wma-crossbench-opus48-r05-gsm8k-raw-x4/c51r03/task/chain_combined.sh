#!/bin/bash
cd /home/ben/task
while ps -p 4144 >/dev/null 2>&1; do sleep 20; done
echo "train done $(date)" >> logs/chain2.log
# copy processor configs into final + checkpoints
for d in runs/sft_combined runs/sft_combined/checkpoint-*; do
  cp "/home/ben/hf_cache/hub/models--google--gemma-3-4b-pt/snapshots/cc012e0a6d0787b4adcc0fa2c4da74402494554d/preprocessor_config.json" "/home/ben/hf_cache/hub/models--google--gemma-3-4b-pt/snapshots/cc012e0a6d0787b4adcc0fa2c4da74402494554d/processor_config.json" "$d/" 2>/dev/null
done
# eval epoch2 (final root)
bash run_eval.sh runs/sft_combined 150 eval_combined_e2 >> logs/chain2.log 2>&1
# eval epoch1 checkpoint
CKPT1=$(ls -d runs/sft_combined/checkpoint-* 2>/dev/null | sort -t- -k2 -n | head -1)
echo "epoch1 ckpt: $CKPT1" >> logs/chain2.log
if [ -n "$CKPT1" ]; then bash run_eval.sh "$CKPT1" 150 eval_combined_e1 >> logs/chain2.log 2>&1; fi
echo "all evals done $(date)" >> logs/chain2.log
