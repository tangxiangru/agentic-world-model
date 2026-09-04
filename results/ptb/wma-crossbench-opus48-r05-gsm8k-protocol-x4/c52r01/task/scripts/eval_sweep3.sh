#!/bin/bash
set -e
export HF_HOME=/home/ben/hf_cache
cd /home/ben/task
SNAP=/home/ben/hf_cache/hub/models--google--gemma-3-4b-pt/snapshots/cc012e0a6d0787b4adcc0fa2c4da74402494554d
mkdir -p eval logs
# assemble eval-ready dirs: copy tokenizer + processor configs into each ckpt dir
for ck in $(ls -d ckpts/exp-03/checkpoint-* 2>/dev/null) ckpts/exp-03/final; do
  [ -d "$ck" ] || continue
  for f in tokenizer.json tokenizer.model tokenizer_config.json special_tokens_map.json added_tokens.json preprocessor_config.json processor_config.json; do
    cp -f $SNAP/$f $ck/ 2>/dev/null || true
  done
done

run_eval () {
  echo "### EVAL $1 -> $2"
  python evaluate.py --model-path "$1" --limit 150 \
    --gpu-memory-utilization 0.6 --max-connections 8 \
    --json-output-file "$2" 2>>logs/eval_sweep3.err
  echo "### DONE $2 : $(cat $2 2>/dev/null | tr -d '\n')"
}

run_eval ckpts/exp-03/checkpoint-464 eval/exp03_checkpoint-464_dev150.json
run_eval ckpts/exp-03/final          eval/exp03_final_dev150.json
echo "ALL DONE"
