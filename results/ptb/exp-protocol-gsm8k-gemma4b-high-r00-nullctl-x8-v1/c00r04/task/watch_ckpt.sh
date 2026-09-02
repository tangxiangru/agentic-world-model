#!/bin/bash
# Preserve selected GRPO checkpoints (weights only) before save_total_limit prunes them.
RUN=${1:-runs/grpo1}
KEEP=${2:-"150 300 450"}
while true; do
  for s in $KEEP; do
    src="$RUN/checkpoint-$s"
    dst="${RUN}_keep_$s"
    if [ -d "$src" ] && [ ! -d "$dst" ]; then
      mkdir -p "$dst"
      cp "$src"/*.safetensors "$src"/*.json "$dst"/ 2>/dev/null
      echo "$(date +%T) kept $s"
    fi
  done
  sleep 45
done
