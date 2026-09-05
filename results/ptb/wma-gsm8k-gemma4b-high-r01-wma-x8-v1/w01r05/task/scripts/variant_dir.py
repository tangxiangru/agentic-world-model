#!/usr/bin/env python3
"""Make a symlink-farm copy of a checkpoint with a different generation_config.json.

vLLM's --generation-config defaults to 'auto', so generation_config.json in the
model directory *is* the served sampling policy (ModelConfig.get_diff_sampling_param
reads temperature/top_k/top_p/repetition_penalty/min_p out of it). This lets a
decode-config experiment be run without copying 8 GB of weights.
"""
from __future__ import annotations

import argparse
import json
import os

ap = argparse.ArgumentParser()
ap.add_argument("--src", required=True)
ap.add_argument("--dst", required=True)
ap.add_argument("--temperature", type=float, default=None)
ap.add_argument("--top-p", type=float, default=None)
ap.add_argument("--top-k", type=int, default=None)
ap.add_argument("--greedy", action="store_true")
a = ap.parse_args()

src = os.path.abspath(a.src)
os.makedirs(a.dst, exist_ok=True)
for f in os.listdir(src):
    if f in ("generation_config.json", "train_summary.json"):
        continue
    d = os.path.join(a.dst, f)
    if not os.path.exists(d):
        os.symlink(os.path.join(src, f), d)

gc = json.load(open(os.path.join(src, "generation_config.json")))
if a.greedy:
    gc["do_sample"] = False
    gc["temperature"] = 0.0
    gc.pop("top_k", None)
    gc.pop("top_p", None)
else:
    if a.temperature is not None:
        gc["temperature"] = a.temperature
    if a.top_p is not None:
        gc["top_p"] = a.top_p
    if a.top_k is not None:
        gc["top_k"] = a.top_k
json.dump(gc, open(os.path.join(a.dst, "generation_config.json"), "w"), indent=2)
print(a.dst, gc)
