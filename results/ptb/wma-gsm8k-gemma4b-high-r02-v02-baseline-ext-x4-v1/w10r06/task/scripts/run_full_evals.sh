#!/bin/bash
set -x
cd /home/ben/task
python evaluate.py --model-path /home/ben/task/ckpts/exp-02-greedy --limit -1 --json-output-file /home/ben/task/eval/exp-06_exp02greedy_full.json > logs/exp-06.log 2>&1
python evaluate.py --model-path /home/ben/task/ckpts/soup-0204-greedy --limit -1 --json-output-file /home/ben/task/eval/exp-06_soup_full.json > logs/exp-06-soup.log 2>&1
python evaluate.py --model-path /home/ben/task/ckpts/exp-04-greedy --limit -1 --json-output-file /home/ben/task/eval/exp-06_exp04_full.json > logs/exp-06-exp04.log 2>&1
echo ALLDONE
