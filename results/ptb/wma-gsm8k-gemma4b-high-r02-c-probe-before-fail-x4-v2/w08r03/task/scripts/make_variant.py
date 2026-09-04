#!/usr/bin/env python3
"""Make a decode-config variant of a checkpoint without copying the weights.

Weights are symlinked, the small json files are copied, and generation_config.json
is rewritten. vLLM reads the decode defaults out of generation_config.json, so this
is the only way to change the graded decode without touching evaluate.py.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil

ap = argparse.ArgumentParser()
ap.add_argument("--src", required=True)
ap.add_argument("--dst", required=True)
ap.add_argument("--greedy", action="store_true")
ap.add_argument("--temperature", type=float, default=None)
ap.add_argument("--top-p", type=float, default=None)
ap.add_argument("--top-k", type=int, default=None)
a = ap.parse_args()

os.makedirs(a.dst, exist_ok=True)
for fn in sorted(os.listdir(a.src)):
    src = os.path.join(a.src, fn)
    dst = os.path.join(a.dst, fn)
    if os.path.isdir(src):
        continue
    if os.path.exists(dst) or os.path.islink(dst):
        os.remove(dst)
    if fn.endswith(".safetensors"):
        os.symlink(os.path.realpath(src), dst)
    else:
        shutil.copy2(src, dst)

gc_path = os.path.join(a.dst, "generation_config.json")
gc = json.load(open(gc_path)) if os.path.exists(gc_path) else {}
if a.greedy:
    # vLLM never reads do_sample: it builds default_sampling_params from
    # generation_config's diff dict, and a missing temperature falls back to 1.0.
    # temperature 0.0 is the only field that actually forces SamplingType.GREEDY
    # (vllm/sampling_params.py then pins top_p=1.0, top_k=0).
    gc["do_sample"] = False
    gc["temperature"] = 0.0
    for k in ("top_k", "top_p"):
        gc.pop(k, None)
else:
    if a.temperature is not None:
        gc["temperature"] = a.temperature
        gc["do_sample"] = True
    if a.top_p is not None:
        gc["top_p"] = a.top_p
    if a.top_k is not None:
        gc["top_k"] = a.top_k
json.dump(gc, open(gc_path, "w"), indent=2)
print(json.dumps(gc, indent=2))
print("wrote", a.dst)
