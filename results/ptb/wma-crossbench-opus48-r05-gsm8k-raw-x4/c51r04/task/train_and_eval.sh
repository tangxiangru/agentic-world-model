#!/bin/bash
set -e
DATA=$1; OUT=$2; EPOCHS=${3:-3}; LR=${4:-1e-5}; BS=${5:-8}; ACCUM=${6:-4}; MAXLEN=${7:-1024}
SNAP=/home/ben/hf_cache/hub/models--google--gemma-3-4b-pt/snapshots/cc012e0a6d0787b4adcc0fa2c4da74402494554d
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python -u train_sft.py --data "$DATA" --out "$OUT" --epochs "$EPOCHS" --bs "$BS" --accum "$ACCUM" --lr "$LR" --maxlen "$MAXLEN" > train_${OUT}.log 2>&1
echo "TRAIN DONE"
cp -L "$SNAP/preprocessor_config.json" "$OUT/"
cp -L "$SNAP/processor_config.json" "$OUT/"
cp greedy_generation_config.json "$OUT/generation_config.json"
echo "POSTPROC DONE"
python evaluate.py --model-path "$OUT" --limit 150 --json-output-file metrics_${OUT}.json --max-connections 4 > eval_${OUT}.log 2>&1
echo "EVAL DONE"
cat metrics_${OUT}.json
