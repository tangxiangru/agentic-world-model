#!/bin/bash
set -u
PID=$1
while kill -0 "$PID" 2>/dev/null; do sleep 20; done
sleep 45
LAST=$(ls -d work/grpo_v3/checkpoint-* 2>/dev/null | sed 's/.*checkpoint-//' | sort -n | tail -1)
if [ -z "$LAST" ]; then echo "no checkpoints from extended run"; exit 1; fi
echo "extended candidate: checkpoint-$LAST"
bash export_and_eval.sh "work/grpo_v3/checkpoint-$LAST" "work/export_ext" "grpo_ext" 500
