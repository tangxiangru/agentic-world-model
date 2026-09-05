#!/bin/bash
set -x
PID=$(cat runs/grpo1.pid)
while ps -p "$PID" > /dev/null 2>&1; do sleep 30; done
sleep 30
bash eval_ckpts.sh runs/grpo1 300 204 272 final
echo "SWEEP DONE"
grep -h . runs/grpo1_204.json runs/grpo1_272.json runs/grpo1_final.json 2>/dev/null
