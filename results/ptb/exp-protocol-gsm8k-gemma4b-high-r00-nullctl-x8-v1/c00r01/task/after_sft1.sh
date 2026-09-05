#!/bin/bash
set -x
while ps -p 3221 > /dev/null 2>&1; do sleep 30; done
sleep 20
export HF_HOME=/home/ben/hf_cache
ls runs/sft1/final || exit 1
python package_model.py runs/sft1/final runs/sft1_greedy --greedy
python package_model.py runs/sft1/final runs/sft1_sample --sampling
bash run_eval.sh runs/sft1_greedy sft1_greedy 200
bash run_eval.sh runs/sft1_sample sft1_sample 200
echo "ALL DONE"
cat runs/sft1_greedy.json runs/sft1_sample.json
