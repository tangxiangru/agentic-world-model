#!/usr/bin/env bash
# Copy the tokenizer/processor files the intermediate Trainer checkpoints do not
# write, so vLLM can load the directory. Usage: prep_ckpt.sh <ckpt_dir>
set -euo pipefail
SNAP=/home/ben/hf_cache/hub/models--google--gemma-3-4b-pt/snapshots/cc012e0a6d0787b4adcc0fa2c4da74402494554d
D="$1"
for f in tokenizer.json tokenizer.model tokenizer_config.json special_tokens_map.json \
         added_tokens.json preprocessor_config.json processor_config.json; do
  [ -e "$D/$f" ] || cp -L "$SNAP/$f" "$D/$f"
done
ls "$D"
