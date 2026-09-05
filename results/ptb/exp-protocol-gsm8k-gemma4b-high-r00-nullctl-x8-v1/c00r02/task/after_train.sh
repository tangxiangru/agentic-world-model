#!/usr/bin/env bash
set -u
while kill -0 4692 2>/dev/null; do sleep 20; done
sleep 30
echo "training exited at $(date)"
cd /home/ben/task
python finalize.py work/sft_v1 --temperature 0
bash run_eval.sh work/sft_v1 300 v1_greedy 32
