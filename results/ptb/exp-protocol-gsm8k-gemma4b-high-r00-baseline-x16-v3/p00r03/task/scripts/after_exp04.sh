#!/bin/bash
set -x
while kill -0 14115 2>/dev/null; do sleep 30; done
sleep 20
cd /home/ben/task
python scripts/finalize_ckpt.py ckpts/exp-04/final ckpts/exp-04/checkpoint-260
python scripts/fast_eval.py --model-path ckpts/exp-04/final --limit 500 --fewshot 10 --out eval/exp04_final_dev500_fs10.json
python scripts/fast_eval.py --model-path ckpts/exp-04/checkpoint-260 --limit 500 --fewshot 10 --out eval/exp04_ck260_dev500_fs10.json
python scripts/fast_eval.py --model-path ckpts/exp-04/final --limit 500 --fewshot 0 --out eval/exp04_final_dev500_fs0.json
python evaluate.py --model-path ckpts/exp-04/final --limit 150 --max-connections 16 --json-output-file eval/exp04_final_official150.json
python evaluate.py --model-path ckpts/exp-04/checkpoint-260 --limit 150 --max-connections 16 --json-output-file eval/exp04_ck260_official150.json
echo "=== ALL DONE ==="
