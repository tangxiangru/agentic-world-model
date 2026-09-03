#!/usr/bin/env bash
set -x
cd /home/ben/task
while pgrep -f "train_sft.py --model .* --out ckpts/exp-02" > /dev/null; do sleep 30; done
sleep 20
ls -la ckpts/exp-02/final
python scripts/finalize_model.py --src ckpts/exp-02/final --dst ckpts/exp-02/hf
bash scripts/run_eval.sh /home/ben/task/ckpts/exp-02/hf exp02_dev150 150 16
