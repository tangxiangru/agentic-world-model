#!/bin/bash
cd /home/ben/task
while ps -p 1996 >/dev/null 2>&1; do sleep 15; done
echo "training finished at $(date)" >> logs/chain.log
sleep 5
bash run_eval.sh runs/sft_gsm 150 eval_sft_gsm >> logs/chain.log 2>&1
echo "eval finished at $(date)" >> logs/chain.log
