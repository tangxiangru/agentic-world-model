#!/bin/bash
# Normalise a training checkpoint into a self-contained model dir and evaluate it.
#   ./export_and_eval.sh <ckpt_dir> <export_dir> <tag> [limit]
set -eu
CKPT=$1
OUT=$2
TAG=$3
LIMIT=${4:-250}
SRC=work/sft_v1   # canonical tokenizer / processor / config source

rm -rf "$OUT"; mkdir -p "$OUT"
cp "$CKPT"/model*.safetensors "$CKPT"/model.safetensors.index.json "$OUT"/
for f in config.json tokenizer.json tokenizer.model tokenizer_config.json \
         special_tokens_map.json added_tokens.json chat_template.jinja \
         preprocessor_config.json processor_config.json; do
  [ -f "$SRC/$f" ] && cp "$SRC/$f" "$OUT"/
done
cp work/generation_config_greedy.json "$OUT"/generation_config.json

python evaluate.py --model-path "$OUT" --limit "$LIMIT" --max-connections 32 \
  --gpu-memory-utilization 0.8 --json-output-file "work/${TAG}.json" \
  > "logs/eval_${TAG}.log" 2>&1
echo "=== $TAG ($CKPT) ==="
cat "work/${TAG}.json"
