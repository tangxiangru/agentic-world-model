#!/bin/bash
cd /home/ben/task
until grep -q "^saved /home/ben/task/ckpts/exp-04/final" logs/exp-04.log; do sleep 20; done
sleep 30
echo "=== training done ==="
python make_variant.py --src ckpts/exp-04/final --dst ckpts/exp-04-greedy --temperature 0.0 --top-p 1.0 --top-k 0 --symlink-weights
python evaluate.py --model-path /home/ben/task/ckpts/exp-04-greedy --limit 150 --max-connections 16 --gpu-memory-utilization 0.85 --json-output-file /home/ben/task/eval/exp-04_greedy_dev150.json > logs/exp-04_eval.log 2>&1
echo "=== dev150 ==="; cat eval/exp-04_greedy_dev150.json
sleep 10
python probe_eval.py --model /home/ben/task/ckpts/exp-04-greedy --temperature 0 --top-p 1.0 --top-k 0 --out analysis/exp-04_probe250.json > logs/exp-04_probe.log 2>&1
echo "=== probe ==="; grep -A6 '"model"' logs/exp-04_probe.log | tail -8
