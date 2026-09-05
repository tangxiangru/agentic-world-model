#!/usr/bin/env bash
set -x
while pgrep -f "train_sft.py --parent /home/ben/task/ckpts/exp-02/served" > /dev/null; do sleep 20; done
sleep 10
python - <<'PY'
import json
p="/home/ben/task/ckpts/exp-02/served/model.safetensors.index.json"
n="/home/ben/task/ckpts/exp-03/final/model.safetensors.index.json"
a=set(json.load(open(p))["weight_map"]); b=set(json.load(open(n))["weight_map"])
print("keys equal:", a==b, "| parent-only:", len(a-b), "| new-only:", len(b-a))
PY
python package_model.py --src /home/ben/task/ckpts/exp-03/final --dst /home/ben/task/ckpts/exp-03/served
bash run_eval.sh /home/ben/task/ckpts/exp-03/served exp-03_n200 200
