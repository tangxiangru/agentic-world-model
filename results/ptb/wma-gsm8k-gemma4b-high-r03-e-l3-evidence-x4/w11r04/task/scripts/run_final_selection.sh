#!/bin/bash
# exp-07: score the two leading candidates on the first 500 official GSM8K test
# items under one identical invocation, so the submission is decided at a
# resolution that can separate a 3 pt gap.
set -x
cd /home/ben/task || exit 1

python evaluate.py --model-path /home/ben/task/ckpts/exp-06/soup \
    --limit 500 --max-connections 16 \
    --json-output-file /home/ben/task/eval/exp-07_exp06soup_n500.json

python evaluate.py --model-path /home/ben/task/ckpts/exp-04/soup \
    --limit 500 --max-connections 16 \
    --json-output-file /home/ben/task/eval/exp-07_exp04soup_n500.json
