#!/bin/bash
cd /home/ben/task
while pgrep -f "train_sft.py --model" > /dev/null; do sleep 20; done
sleep 20
echo "=== training done, starting dev150 eval ==="
python evaluate.py --model-path /home/ben/task/ckpts/exp-02/final --limit 150 --max-connections 16 --gpu-memory-utilization 0.85 --json-output-file /home/ben/task/eval/exp-02_dev150.json > /home/ben/task/logs/exp-02_eval.log 2>&1
echo "=== dev150 done ==="
cat /home/ben/task/eval/exp-02_dev150.json
sleep 10
python probe_eval.py --model /home/ben/task/ckpts/exp-02/final --out /home/ben/task/analysis/exp-02_probe250.json > /home/ben/task/logs/exp-02_probe.log 2>&1
echo "=== probe done ==="
tail -20 /home/ben/task/logs/exp-02_probe.log
