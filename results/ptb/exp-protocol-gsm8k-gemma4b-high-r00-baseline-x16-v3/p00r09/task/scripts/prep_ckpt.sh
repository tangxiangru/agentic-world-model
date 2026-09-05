#!/usr/bin/env bash
# Trainer's intermediate checkpoint-N dirs have no tokenizer; vLLM needs one.
set -euo pipefail
SRC="$1"
BASE=/home/ben/hf_cache/hub/models--google--gemma-3-4b-pt/snapshots/cc012e0a6d0787b4adcc0fa2c4da74402494554d
for f in tokenizer.json tokenizer_config.json tokenizer.model special_tokens_map.json \
         added_tokens.json preprocessor_config.json processor_config.json generation_config.json; do
  [ -f "$SRC/$f" ] || cp "$BASE/$f" "$SRC/$f"
done
echo "prepared $SRC"
