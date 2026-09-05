#!/bin/bash
set -x
while kill -0 9251 2>/dev/null; do sleep 30; done
sleep 20
cd /home/ben/task
python scripts/finalize_ckpt.py ckpts/exp-03/final
python scripts/fast_eval.py --model-path ckpts/exp-03/final --limit 500 --fewshot 10 --out eval/exp03_final_dev500_fs10.json
python scripts/fast_eval.py --model-path ckpts/exp-03/final --limit 500 --fewshot 0 --out eval/exp03_final_dev500_fs0.json
python evaluate.py --model-path ckpts/exp-03/final --limit 150 --max-connections 16 --json-output-file eval/exp03_final_official150.json
echo "=== ALL DONE ==="
