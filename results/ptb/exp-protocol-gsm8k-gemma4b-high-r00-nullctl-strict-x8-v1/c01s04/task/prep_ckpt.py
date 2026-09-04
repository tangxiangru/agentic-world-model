#!/usr/bin/env python3
"""Materialise a Trainer checkpoint dir as a vLLM-loadable model dir (greedy decoding)."""
import argparse
import json
import os
import shutil
import sys

BASE = os.environ["PTB_BASE_MODEL_SNAPSHOT"]

ap = argparse.ArgumentParser()
ap.add_argument("src")
ap.add_argument("dst")
ap.add_argument("--sampling", action="store_true")
a = ap.parse_args()

os.makedirs(a.dst, exist_ok=True)
for f in os.listdir(a.src):
    if f.endswith(".safetensors"):
        d = os.path.join(a.dst, f)
        if os.path.islink(d) or os.path.exists(d):
            os.remove(d)
        os.symlink(os.path.abspath(os.path.join(a.src, f)), d)
    elif f.endswith(".json") or f.endswith(".jinja") or f.endswith(".model"):
        shutil.copy(os.path.join(a.src, f), os.path.join(a.dst, f))
for f in ["preprocessor_config.json", "processor_config.json", "tokenizer.model"]:
    d = os.path.join(a.dst, f)
    if not os.path.exists(d):
        shutil.copy(os.path.join(BASE, f), d)

gc = {"bos_token_id": 2, "cache_implementation": "hybrid", "eos_token_id": [1, 106], "pad_token_id": 0}
if a.sampling:
    gc.update({"do_sample": True, "top_k": 64, "top_p": 0.95})
else:
    gc.update({"do_sample": False, "temperature": 0.0})
json.dump(gc, open(os.path.join(a.dst, "generation_config.json"), "w"), indent=2)
print("prepared", a.dst, sorted(os.listdir(a.dst)))
