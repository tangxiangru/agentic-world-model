#!/usr/bin/env python3
"""Clone a checkpoint dir (hardlinked weights, so ~free) with a different generation_config.

vLLM 0.11 reads generation_config.json out of the model directory and uses it as the
default sampling params ("Default sampling parameters have been overridden by the model's
Hugging Face generation config", logs/exp-01.log). evaluate.py passes no temperature, so
whatever sits in that file IS the decoding policy at grading time. The base repo ships
do_sample=true / top_k=64 / top_p=0.95, i.e. temperature-1.0 sampling.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil

ap = argparse.ArgumentParser()
ap.add_argument("--src", required=True)
ap.add_argument("--dst", required=True)
ap.add_argument("--mode", choices=["greedy", "asis"], default="greedy")
a = ap.parse_args()

os.makedirs(a.dst, exist_ok=True)
for f in sorted(os.listdir(a.src)):
    s, d = os.path.join(a.src, f), os.path.join(a.dst, f)
    if os.path.exists(d):
        os.remove(d)
    if f.endswith(".safetensors"):
        os.link(s, d)          # hardlink: no extra disk, no copy time
    elif os.path.isfile(s):
        shutil.copy(s, d)

gc_path = os.path.join(a.dst, "generation_config.json")
gc = json.load(open(gc_path)) if os.path.exists(gc_path) else {}
if a.mode == "greedy":
    gc["do_sample"] = False
    gc["temperature"] = 0.0
    gc.pop("top_k", None)
    gc.pop("top_p", None)
json.dump(gc, open(gc_path, "w"), indent=2)
print(json.dumps({"dst": a.dst, "generation_config": gc}, indent=2))
