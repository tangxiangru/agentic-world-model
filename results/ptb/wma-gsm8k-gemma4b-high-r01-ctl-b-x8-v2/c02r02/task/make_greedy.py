#!/usr/bin/env python3
"""Copy a checkpoint and rewrite generation_config.json for greedy decoding.

vLLM reads generation_config.json and uses it as the default SamplingParams for
every request ("Default sampling parameters have been overridden by the model's
Hugging Face generation config"). The stock gemma-3 file asks for do_sample with
top_k=64 / top_p=0.95 and no temperature, which makes the grader score a
temperature-1.0 sample. This writes temperature 0 and drops the truncation
params so vLLM decodes greedily. Weights are untouched.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil

ap = argparse.ArgumentParser()
ap.add_argument("--src", required=True)
ap.add_argument("--dst", required=True)
ap.add_argument("--link", action="store_true", help="hardlink the weights instead of copying")
a = ap.parse_args()

os.makedirs(a.dst, exist_ok=True)
for fn in sorted(os.listdir(a.src)):
    s, d = os.path.join(a.src, fn), os.path.join(a.dst, fn)
    if os.path.exists(d):
        os.remove(d)
    if a.link and fn.endswith(".safetensors"):
        os.link(s, d)
    else:
        shutil.copy(s, d)

p = os.path.join(a.dst, "generation_config.json")
g = json.load(open(p))
before = dict(g)
g["do_sample"] = False
g["temperature"] = 0.0
g.pop("top_k", None)
g.pop("top_p", None)
json.dump(g, open(p, "w"), indent=2)
print("before:", json.dumps(before))
print("after :", json.dumps(g))
print("wrote", a.dst)
