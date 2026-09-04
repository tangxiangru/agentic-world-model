#!/usr/bin/env bash
set -x
while pgrep -f "train_sft.py --parent /home/ben/task/ckpts/exp-03/served" > /dev/null; do sleep 20; done
sleep 10
python - <<'PY'
import json
a=set(json.load(open("/home/ben/task/ckpts/exp-03/served/model.safetensors.index.json"))["weight_map"])
b=set(json.load(open("/home/ben/task/ckpts/exp-04/final/model.safetensors.index.json"))["weight_map"])
print("keys equal:", a==b, "| parent-only:", len(a-b), "| new-only:", len(b-a))
PY
python package_model.py --src /home/ben/task/ckpts/exp-04/final --dst /home/ben/task/ckpts/exp-04/served
bash run_eval.sh /home/ben/task/ckpts/exp-04/served exp-04_n200 200
