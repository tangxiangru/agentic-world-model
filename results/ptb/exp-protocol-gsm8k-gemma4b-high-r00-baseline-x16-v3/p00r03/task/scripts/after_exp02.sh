#!/bin/bash
# Waits for the exp-02 trainer to exit, then runs the card's diagnostics and the
# official protocol. Sequential so only one process ever holds the GPU.
set -x
while kill -0 3946 2>/dev/null; do sleep 30; done
sleep 20

cd /home/ben/task
python scripts/finalize_ckpt.py ckpts/exp-02/final ckpts/exp-02/checkpoint-900 ckpts/exp-02/checkpoint-1800

python scripts/fast_eval.py --model-path ckpts/exp-02/final --limit 500 --fewshot 10 \
  --out eval/exp02_final_dev500_fs10.json
python scripts/fast_eval.py --model-path ckpts/exp-02/final --limit 500 --fewshot 0 \
  --out eval/exp02_final_dev500_fs0.json
python scripts/fast_eval.py --model-path ckpts/exp-02/checkpoint-900 --limit 500 --fewshot 10 \
  --out eval/exp02_ck900_dev500_fs10.json

python evaluate.py --model-path ckpts/exp-02/final --limit 150 --max-connections 16 \
  --json-output-file eval/exp02_final_official150.json
echo "=== ALL DONE ==="
