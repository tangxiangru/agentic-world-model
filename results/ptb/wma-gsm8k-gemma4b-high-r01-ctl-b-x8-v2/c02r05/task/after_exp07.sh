#!/bin/bash
cd /home/ben/task
until grep -q "^saved /home/ben/task/ckpts/exp-07/final" logs/exp-07.log; do sleep 20; done
sleep 30
python make_variant.py --src ckpts/exp-07/final --dst ckpts/exp-07-greedy --temperature 0.0 --top-p 1.0 --top-k 0 --symlink-weights > /dev/null
python probe_eval.py --model /home/ben/task/ckpts/exp-07-greedy --temperature 0 --top-p 1.0 --top-k 0 --out analysis/exp-07_probe250.json > logs/exp-07_probe.log 2>&1
echo "=== probe ==="; grep -A6 '"model"' logs/exp-07_probe.log | tail -8
sleep 10
python evaluate.py --model-path /home/ben/task/ckpts/exp-07-greedy --limit 500 --max-connections 16 --gpu-memory-utilization 0.85 --json-output-file /home/ben/task/eval/exp-07_greedy_dev500.json > logs/exp-07_eval.log 2>&1
echo "=== dev500 ==="; cat eval/exp-07_greedy_dev500.json
