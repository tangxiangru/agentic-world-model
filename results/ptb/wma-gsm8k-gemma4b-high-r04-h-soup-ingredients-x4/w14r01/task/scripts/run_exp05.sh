#!/usr/bin/env bash
# exp-05: uniform weight average of the three stage-2 checkpoints, read greedily.
set -eu
cd /home/ben/task
OUT=/home/ben/task/ckpts/exp-05_soup
rm -rf "$OUT"
python scripts/soup.py \
    --ckpts /home/ben/task/ckpts/exp-04/checkpoint-742 \
            /home/ben/task/ckpts/exp-04/checkpoint-1484 \
            /home/ben/task/ckpts/exp-04/checkpoint-2226 \
    --out "$OUT"
python scripts/set_decode.py --ckpt "$OUT" --mode greedy
INSPECT_LOG_DIR=/home/ben/task/eval/exp-05_soup_dev150_logs python evaluate.py \
    --model-path "$OUT" \
    --limit 150 \
    --json-output-file /home/ben/task/eval/exp-05_soup_dev150.json \
    --max-connections 2 \
    --max-tokens 4000 \
    --gpu-memory-utilization 0.3
cat /home/ben/task/eval/exp-05_soup_dev150.json
python scripts/analyze_eval.py --log-dir /home/ben/task/eval/exp-05_soup_dev150_logs \
    --out /home/ben/task/analysis/exp-05_soup_diag.json
