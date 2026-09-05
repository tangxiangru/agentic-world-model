#!/bin/bash
cd /home/ben/task
# wait for training to finish (final dir written and train.py gone)
while pgrep -f "train.py --data data/exp04_train.jsonl" >/dev/null; do sleep 20; done
sleep 10
# patch generation_config to proven greedy (train.py saves temperature=null)
python - <<'PY'
import json
cfg={"bos_token_id":2,"cache_implementation":"hybrid","do_sample":False,
     "eos_token_id":[1,106],"pad_token_id":0,"temperature":0.0,
     "transformers_version":"4.57.3"}
json.dump(cfg, open("ckpts/exp04/final/generation_config.json","w"), indent=2)
print("patched generation_config:", json.load(open("ckpts/exp04/final/generation_config.json")))
PY
# free GPU
for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do kill -9 "$p" 2>/dev/null; done
sleep 5
# full-test eval
python evaluate.py --model-path ckpts/exp04/final --limit 1319 --max-connections 32 \
  --gpu-memory-utilization 0.85 --json-output-file eval/exp04_full.json
echo "EXP04_EVAL_DONE"
cat eval/exp04_full.json
