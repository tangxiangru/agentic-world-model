#!/usr/bin/env bash
# Evaluate several checkpoints under the identical locked protocol, one after
# the other, and write one json of metrics + one inspect log dir per checkpoint.
#   usage: sweep_eval.sh <tag> <limit> <ckpt> [<ckpt> ...]
set -u
cd /home/ben/task
TAG=$1; shift
LIMIT=$1; shift
for CKPT in "$@"; do
  NAME=$(basename "$CKPT")
  OUT="eval/${TAG}_${NAME}_dev${LIMIT}"
  if [ -f "${OUT}.json" ]; then echo "skip $NAME (already scored)"; continue; fi
  echo "=== $(date -u +%H:%M:%S) evaluating $CKPT -> ${OUT}.json"
  INSPECT_LOG_DIR="/home/ben/task/${OUT}_logs" python evaluate.py \
      --model-path "$CKPT" \
      --limit "$LIMIT" \
      --json-output-file "/home/ben/task/${OUT}.json" \
      --max-connections 2 \
      --max-tokens 4000 \
      --gpu-memory-utilization 0.3 \
      > "logs/${TAG}_${NAME}.log" 2>&1
  echo "exit=$? $(cat ${OUT}.json 2>/dev/null | tr -d '\n')"
  python scripts/analyze_eval.py --log-dir "/home/ben/task/${OUT}_logs" \
      --out "analysis/${TAG}_${NAME}_diag.json" 2>/dev/null | head -20
done
echo "=== sweep done $(date -u +%H:%M:%S)"
