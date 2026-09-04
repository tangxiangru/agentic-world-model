#!/bin/bash
# Prepare (tokenizer + processor + greedy gen config) and greedy-eval each exp-04
# checkpoint at limit 150, sequentially, detached.
cd /home/ben/task
BASE=/home/ben/hf_cache/hub/models--google--gemma-3-4b-pt/snapshots/cc012e0a6d0787b4adcc0fa2c4da74402494554d
TOK_FILES="added_tokens.json special_tokens_map.json tokenizer.json tokenizer.model tokenizer_config.json"
PROC_FILES="preprocessor_config.json processor_config.json"
for ck in $(ls -d ckpts/exp-04/checkpoint-* 2>/dev/null | sort -t- -k3 -n); do
  d="$ck"
  for f in $TOK_FILES; do [ -f "$d/$f" ] || cp "$BASE/$f" "$d/$f"; done
  for f in $PROC_FILES; do [ -f "$d/$f" ] || cp "$BASE/$f" "$d/$f"; done
  python scripts/set_greedy.py "$d/generation_config.json" >/dev/null
  name=$(basename "$ck")
  out="eval/exp04_${name}_greedy_dev150.json"
  echo "=== evaluating $name -> $out ==="
  python evaluate.py --model-path "$d" --limit 150 --max-connections 8 \
    --json-output-file "$out" > "logs/eval_exp04_${name}_greedy.log" 2>&1
  echo "$name greedy exit=$? result=$(cat $out 2>/dev/null | tr -d '\n')"
done
echo "EVAL_EXP04_DONE"
