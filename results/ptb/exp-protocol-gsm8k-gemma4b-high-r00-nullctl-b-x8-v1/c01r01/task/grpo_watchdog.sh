#!/bin/bash
# Stop the GRPO run after a fixed wall-clock budget so there is always time left
# to evaluate checkpoints and export final_model.
set -u
PID=$1
SECS=$2
sleep "$SECS"
if kill -0 "$PID" 2>/dev/null; then
  echo "$(date) stopping GRPO pid $PID"
  kill -INT "$PID" 2>/dev/null
  sleep 90
  kill -9 "$PID" 2>/dev/null
fi
sleep 30
echo "GRPO stopped. checkpoints:"
ls -d work/grpo_v2/checkpoint-* 2>/dev/null
