#!/usr/bin/env bash
# exp-03: same weights as ckpts/exp-02/final, greedy decode arm.
set -eu
cd /home/ben/task
SRC=/home/ben/task/ckpts/exp-02/final
DST=/home/ben/task/ckpts/exp-03_greedy
rm -rf "$DST"
mkdir -p "$DST"
# hard-link the weights so the arm is the ONLY difference and no disk is spent
for f in "$SRC"/*; do ln -f "$f" "$DST/$(basename "$f")"; done
rm -f "$DST/generation_config.json"
cp "$SRC/generation_config.json" "$DST/generation_config.json"
python scripts/set_decode.py --ckpt "$DST" --mode greedy
INSPECT_LOG_DIR=/home/ben/task/eval/exp-03_greedy_dev150_logs python evaluate.py \
    --model-path "$DST" \
    --limit 150 \
    --json-output-file /home/ben/task/eval/exp-03_greedy_dev150.json \
    --max-connections 2 \
    --max-tokens 4000 \
    --gpu-memory-utilization 0.3
cat /home/ben/task/eval/exp-03_greedy_dev150.json
python scripts/analyze_eval.py --log-dir /home/ben/task/eval/exp-03_greedy_dev150_logs \
    --out /home/ben/task/analysis/exp-03_greedy_diag.json
