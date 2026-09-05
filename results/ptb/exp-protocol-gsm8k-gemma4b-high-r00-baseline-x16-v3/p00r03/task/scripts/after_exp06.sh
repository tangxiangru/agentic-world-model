#!/bin/bash
set -x
while kill -0 21224 2>/dev/null; do sleep 20; done
sleep 15
cd /home/ben/task
python scripts/finalize_ckpt.py ckpts/exp-06/final
python evaluate.py --model-path ckpts/exp-06/final --limit 500 --max-connections 16 --json-output-file eval/exp06_final_official500.json
python scripts/fast_eval.py --model-path ckpts/exp-06/final --limit 500 --fewshot 10 --out eval/exp06_final_dev500_fs10.json
python scripts/fast_eval.py --model-path ckpts/exp-06/final --limit 500 --fewshot 0 --out eval/exp06_final_dev500_fs0.json
echo "=== ALL DONE ==="
