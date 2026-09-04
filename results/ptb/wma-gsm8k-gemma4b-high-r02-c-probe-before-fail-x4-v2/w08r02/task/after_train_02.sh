#!/usr/bin/env bash
set -x
# wait for the training process to exit
while pgrep -f "train_sft.py --data data/sft_v1.jsonl --out /home/ben/task/ckpts/exp-02" > /dev/null; do sleep 20; done
sleep 10
ls -la /home/ben/task/ckpts/exp-02/
python - <<'PY'
import json, os
base="/home/ben/hf_cache/hub/models--google--gemma-3-4b-pt/snapshots/cc012e0a6d0787b4adcc0fa2c4da74402494554d"
new="/home/ben/task/ckpts/exp-02/final"
b=set(json.load(open(base+"/model.safetensors.index.json"))["weight_map"])
n=set(json.load(open(new+"/model.safetensors.index.json"))["weight_map"])
print("keys equal:", b==n, "| base-only:", len(b-n), "| new-only:", len(n-b))
for k in list(b-n)[:5]: print("  base-only", k)
for k in list(n-b)[:5]: print("  new-only ", k)
PY
python package_model.py --src /home/ben/task/ckpts/exp-02/final --dst /home/ben/task/ckpts/exp-02/served
bash run_eval.sh /home/ben/task/ckpts/exp-02/served exp-02_n200 200
