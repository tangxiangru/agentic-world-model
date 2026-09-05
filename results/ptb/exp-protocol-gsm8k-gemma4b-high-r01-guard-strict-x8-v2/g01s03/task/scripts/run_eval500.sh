#!/bin/bash
cd /home/ben/task
python evaluate.py --model-path /home/ben/task/ckpts/exp-05/final --limit 500 --max-connections 16 --json-output-file /home/ben/task/eval/exp-06_exp05_dev500.json > logs/exp-06_exp05_eval500.log 2>&1
sleep 20
python evaluate.py --model-path /home/ben/task/ckpts/exp-04/final --limit 500 --max-connections 16 --json-output-file /home/ben/task/eval/exp-06_exp04_dev500.json > logs/exp-06_exp04_eval500.log 2>&1
echo DONE_BOTH
