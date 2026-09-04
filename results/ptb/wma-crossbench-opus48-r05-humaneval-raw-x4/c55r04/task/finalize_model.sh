#!/bin/bash
# Copy processor/preprocessor configs from base snapshot into a trained model dir
# so vllm can load the Gemma3ForConditionalGeneration model.
set -e
SRC=/home/ben/hf_cache/hub/models--google--gemma-3-4b-pt/snapshots/cc012e0a6d0787b4adcc0fa2c4da74402494554d
DST="$1"
for f in preprocessor_config.json processor_config.json; do
  if [ ! -f "$DST/$f" ] && [ -f "$SRC/$f" ]; then
    cp "$SRC/$f" "$DST/$f"
    echo "copied $f"
  fi
done
echo "finalized $DST"
ls -la "$DST"
