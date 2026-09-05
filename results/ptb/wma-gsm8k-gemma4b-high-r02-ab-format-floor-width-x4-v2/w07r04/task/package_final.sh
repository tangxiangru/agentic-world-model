#!/usr/bin/env bash
# Copy the chosen checkpoint into final_model/, pin the greedy decode config,
# and verify it loads the way the grader will load it.
# usage: bash package_final.sh <checkpoint-dir>
set -euo pipefail
SRC="$1"
rm -rf final_model
cp -r "$SRC" final_model
python fix_gen_config.py final_model
python - <<'PY'
import json, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
cfg = json.load(open("final_model/config.json"))
print("architectures:", cfg["architectures"])
m = AutoModelForCausalLM.from_pretrained("final_model", dtype=torch.bfloat16)
t = AutoTokenizer.from_pretrained("final_model")
print("CPU load OK:", type(m).__name__, f"{sum(p.numel() for p in m.parameters())/1e9:.2f}B")
print("generation_config:", json.load(open("final_model/generation_config.json")))
print("tokenizer files present:", t.vocab_size)
PY
ls -la final_model
