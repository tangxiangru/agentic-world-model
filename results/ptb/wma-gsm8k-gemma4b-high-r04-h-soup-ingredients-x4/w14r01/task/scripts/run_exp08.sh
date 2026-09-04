#!/usr/bin/env bash
# exp-08: four-way uniform weight average across BOTH stage-2 trajectories.
set -eu
cd /home/ben/task
OUT=/home/ben/task/ckpts/exp-08_soup4
rm -rf "$OUT"
python scripts/soup.py \
    --ckpts /home/ben/task/ckpts/exp-04/checkpoint-742 \
            /home/ben/task/ckpts/exp-04/checkpoint-1484 \
            /home/ben/task/ckpts/exp-04/checkpoint-2226 \
            /home/ben/task/ckpts/exp-07/final \
    --out "$OUT"
python scripts/set_decode.py --ckpt "$OUT" --mode greedy
INSPECT_LOG_DIR=/home/ben/task/eval/exp-08_soup4_full1319_logs python evaluate.py \
    --model-path "$OUT" --limit -1 \
    --json-output-file /home/ben/task/eval/exp-08_soup4_full1319.json \
    --max-connections 2 --max-tokens 4000 --gpu-memory-utilization 0.3
cat /home/ben/task/eval/exp-08_soup4_full1319.json
python scripts/analyze_eval.py --log-dir /home/ben/task/eval/exp-08_soup4_full1319_logs \
    --out /home/ben/task/analysis/exp-08_soup4_diag.json
