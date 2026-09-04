#!/usr/bin/env bash
# exp-06: decide the submission on the FULL gsm8k test set (1319 items) instead
# of the 150 items that already selected the candidates.
set -u
cd /home/ben/task
for CKPT in "$@"; do
  NAME=$(basename "$(dirname "$CKPT")")_$(basename "$CKPT")
  OUT="eval/exp-06_${NAME}_full1319"
  if [ -f "${OUT}.json" ]; then echo "skip $NAME"; continue; fi
  echo "=== $(date -u +%H:%M:%S) full read: $CKPT"
  INSPECT_LOG_DIR="/home/ben/task/${OUT}_logs" python evaluate.py \
      --model-path "$CKPT" \
      --limit -1 \
      --json-output-file "/home/ben/task/${OUT}.json" \
      --max-connections 2 \
      --max-tokens 4000 \
      --gpu-memory-utilization 0.3 \
      > "logs/exp-06_${NAME}.log" 2>&1
  echo "exit=$? $(tr -d '\n' < ${OUT}.json 2>/dev/null)"
  python scripts/analyze_eval.py --log-dir "/home/ben/task/${OUT}_logs" \
      --out "analysis/exp-06_${NAME}_diag.json" 2>/dev/null | head -18
done
echo "=== exp-06 done $(date -u +%H:%M:%S)"
