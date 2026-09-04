#!/bin/bash
# usage: bash eval_model.sh <model_path> <limit> <max_connections>
MP=${1:-sft_v2}
LIM=${2:-100}
MC=${3:-6}
OUT="metrics_$(basename $MP)_lim${LIM}.json"
SNAP=/home/ben/hf_cache/hub/models--google--gemma-3-4b-pt/snapshots/cc012e0a6d0787b4adcc0fa2c4da74402494554d
for f in preprocessor_config.json processor_config.json; do
  if [ ! -f "$MP/$f" ] && [ -d "$MP" ]; then cp "$SNAP/$f" "$MP/" && echo "auto-copied $f into $MP"; fi
done
echo "Evaluating $MP limit=$LIM -> $OUT"
python evaluate.py \
  --model-path "$MP" \
  --limit "$LIM" \
  --max-connections "$MC" \
  --json-output-file "$OUT" > "eval_$(basename $MP)_lim${LIM}.log" 2>&1
echo "EXIT=$?"
echo "=== METRICS ==="
cat "$OUT" 2>/dev/null
echo
echo "=== log tail ==="
tail -15 "eval_$(basename $MP)_lim${LIM}.log"
