#!/bin/bash
# Usage: bash eval_ckpt.sh <model_dir> <limit> <out_json>
set -e
DIR="$1"; LIMIT="${2:-150}"; OUT="${3:-eval_tmp.json}"
SNAP="$PTB_BASE_MODEL_SNAPSHOT"
# ensure multimodal preprocessing configs exist so vLLM can load the model
for f in preprocessor_config.json processor_config.json; do
  if [ ! -f "$DIR/$f" ] && [ -f "$SNAP/$f" ]; then cp "$SNAP/$f" "$DIR/$f"; fi
done
python evaluate.py --model-path "$DIR" --limit "$LIMIT" --json-output-file "$OUT" --max-connections 4
echo "=== RESULT ($DIR) ==="; cat "$OUT"
