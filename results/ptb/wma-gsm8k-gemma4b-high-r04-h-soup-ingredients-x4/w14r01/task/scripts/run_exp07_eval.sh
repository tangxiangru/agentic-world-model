#!/usr/bin/env bash
# exp-07 evaluation: greedy arm, full 1319-item test set.
# train_sft.py copies the PARENT's generation_config into every checkpoint, and
# exp-02/final's is the sampling one - so the arm must be set explicitly here.
set -u
cd /home/ben/task
for CKPT in "$@"; do
  python scripts/set_decode.py --ckpt "$CKPT" --mode greedy > /dev/null
  NAME=exp-07_$(basename "$CKPT")
  OUT="eval/${NAME}_full1319"
  echo "=== $(date -u +%H:%M:%S) full read: $CKPT"
  INSPECT_LOG_DIR="/home/ben/task/${OUT}_logs" python evaluate.py \
      --model-path "$CKPT" --limit -1 \
      --json-output-file "/home/ben/task/${OUT}.json" \
      --max-connections 2 --max-tokens 4000 --gpu-memory-utilization 0.3 \
      > "logs/${NAME}.log" 2>&1
  echo "exit=$? $(tr -d '\n' < ${OUT}.json 2>/dev/null)"
  python scripts/analyze_eval.py --log-dir "/home/ben/task/${OUT}_logs" \
      --out "analysis/${NAME}_diag.json" 2>/dev/null | head -18
done
echo "=== exp-07 eval done $(date -u +%H:%M:%S)"
