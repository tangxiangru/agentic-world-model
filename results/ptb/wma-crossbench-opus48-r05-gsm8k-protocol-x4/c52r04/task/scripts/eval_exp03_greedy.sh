#!/bin/bash
# Prepare (tokenizer + processor + greedy gen config) and greedy-eval each exp-03
# step checkpoint at limit 150, sequentially, fully detached.
cd /home/ben/task
BASE=/home/ben/hf_cache/hub/models--google--gemma-3-4b-pt/snapshots/cc012e0a6d0787b4adcc0fa2c4da74402494554d
TOK_FILES="added_tokens.json special_tokens_map.json tokenizer.json tokenizer.model tokenizer_config.json"
PROC_FILES="preprocessor_config.json processor_config.json"

for ck in checkpoint-120 checkpoint-240 checkpoint-360 checkpoint-480 checkpoint-600 checkpoint-720 checkpoint-840 checkpoint-858; do
  d="ckpts/exp-03/$ck"
  [ -d "$d" ] || { echo "MISSING $d, skipping"; continue; }
  for f in $TOK_FILES; do [ -f "$d/$f" ] || cp "$BASE/$f" "$d/$f"; done
  for f in $PROC_FILES; do [ -f "$d/$f" ] || cp "$BASE/$f" "$d/$f"; done
  python scripts/set_greedy.py "$d/generation_config.json" >/dev/null
  out="eval/exp03_${ck}_greedy_dev150.json"
  echo "=== evaluating $ck -> $out ==="
  python evaluate.py --model-path "$d" --limit 150 --max-connections 8 \
    --json-output-file "$out" > "logs/eval_exp03_${ck}_greedy.log" 2>&1
  echo "$ck greedy exit=$? result=$(cat $out 2>/dev/null | tr -d '\n')"
done
echo "EVAL_EXP03_DONE"
