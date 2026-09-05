#!/bin/bash
# Prepare (tokenizer + processor configs + greedy gen config) and greedy-eval each
# exp-02 epoch checkpoint at limit 150. Runs sequentially, fully detached.
cd /home/ben/task
BASE=/home/ben/hf_cache/hub/models--google--gemma-3-4b-pt/snapshots/cc012e0a6d0787b4adcc0fa2c4da74402494554d
TOK_FILES="added_tokens.json special_tokens_map.json tokenizer.json tokenizer.model tokenizer_config.json"
PROC_FILES="preprocessor_config.json processor_config.json"

for ck in checkpoint-728 checkpoint-1456 checkpoint-2184; do
  d="ckpts/exp-02/$ck"
  [ -d "$d" ] || { echo "MISSING $d, skipping"; continue; }
  # copy tokenizer files if absent
  for f in $TOK_FILES; do [ -f "$d/$f" ] || cp "$BASE/$f" "$d/$f"; done
  # copy processor configs if absent
  for f in $PROC_FILES; do [ -f "$d/$f" ] || cp "$BASE/$f" "$d/$f"; done
  # set greedy
  python scripts/set_greedy.py "$d/generation_config.json"
  out="eval/exp02_${ck}_greedy_dev150.json"
  echo "=== evaluating $ck -> $out ==="
  python evaluate.py --model-path "$d" --limit 150 --max-connections 8 \
    --json-output-file "$out" > "logs/eval_exp02_${ck}_greedy.log" 2>&1
  echo "$ck greedy exit=$? result=$(cat $out 2>/dev/null | tr -d '\n')"
done
echo "EVAL_EXP02_DONE"
