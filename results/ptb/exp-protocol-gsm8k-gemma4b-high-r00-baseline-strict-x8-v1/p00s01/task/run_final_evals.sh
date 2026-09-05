#!/bin/bash
cd /home/ben/task
set -x
python evaluate.py --model-path /home/ben/task/ckpts/exp-05/final --limit 500 --max-connections 16 --json-output-file /home/ben/task/eval/exp05_dev500.json
python evaluate.py --model-path /home/ben/task/ckpts/exp-06/final --limit 500 --max-connections 16 --json-output-file /home/ben/task/eval/exp06_dev500.json
python evaluate.py --model-path /home/ben/task/ckpts/exp-04/final --limit 500 --max-connections 16 --json-output-file /home/ben/task/eval/exp04_dev500.json
