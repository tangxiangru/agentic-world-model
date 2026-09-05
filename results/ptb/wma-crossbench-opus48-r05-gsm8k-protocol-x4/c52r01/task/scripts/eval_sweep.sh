#!/bin/bash
# Evaluate base (comparator) + each epoch checkpoint on dev-150 under one protocol.
set -e
export HF_HOME=/home/ben/hf_cache
cd /home/ben/task
SNAP=/home/ben/hf_cache/hub/models--google--gemma-3-4b-pt/snapshots/cc012e0a6d0787b4adcc0fa2c4da74402494554d
FINAL=ckpts/exp-01/final

# make epoch checkpoints eval-ready (copy tokenizer + processor configs)
for ck in ckpts/exp-01/checkpoint-234 ckpts/exp-01/checkpoint-468; do
  for f in tokenizer.json tokenizer.model tokenizer_config.json special_tokens_map.json added_tokens.json preprocessor_config.json processor_config.json; do
    cp -f $FINAL/$f $ck/ 2>/dev/null || true
  done
done

run_eval () {
  local path="$1"; local out="$2"
  echo "### EVAL $path -> $out"
  python evaluate.py --model-path "$path" --limit 150 \
    --gpu-memory-utilization 0.6 --max-connections 8 \
    --json-output-file "$out" 2>>logs/eval_sweep.err
  echo "### DONE $out : $(cat $out 2>/dev/null)"
}

run_eval "$SNAP"                    eval/base_dev150.json
run_eval ckpts/exp-01/checkpoint-234 eval/exp01_ep1_dev150.json
run_eval ckpts/exp-01/checkpoint-468 eval/exp01_ep2_dev150.json
run_eval "$FINAL"                    eval/exp01_ep3_dev150.json
echo "ALL DONE"
