#!/bin/bash
cd /home/ben/task
while ps -p 10249 >/dev/null 2>&1; do sleep 20; done
echo "c2 train done $(date)" >> logs/chain_c2.log
for d in runs/sft_c2 runs/sft_c2/checkpoint-*; do
  cp "/home/ben/hf_cache/hub/models--google--gemma-3-4b-pt/snapshots/cc012e0a6d0787b4adcc0fa2c4da74402494554d/preprocessor_config.json" "/home/ben/hf_cache/hub/models--google--gemma-3-4b-pt/snapshots/cc012e0a6d0787b4adcc0fa2c4da74402494554d/processor_config.json" "$d/" 2>/dev/null
done
CKPTS=$(ls -d runs/sft_c2/checkpoint-* 2>/dev/null | sort -t- -k2 -n)
i=1
for c in $CKPTS; do
  echo "eval epoch$i: $c" >> logs/chain_c2.log
  bash run_eval.sh "$c" 150 eval_c2_e$i >> logs/chain_c2.log 2>&1
  i=$((i+1))
done
echo "eval epoch3(root): runs/sft_c2" >> logs/chain_c2.log
bash run_eval.sh runs/sft_c2 150 eval_c2_e3 >> logs/chain_c2.log 2>&1
echo "c2 all evals done $(date)" >> logs/chain_c2.log
