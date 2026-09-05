#!/usr/bin/env python3
"""Uniform weight average ("model soup") of two checkpoints fine-tuned from the
same base snapshot, saved as a normal Gemma3ForConditionalGeneration checkpoint
with greedy generation_config.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil

import torch
from transformers import AutoTokenizer, Gemma3ForConditionalGeneration

ap = argparse.ArgumentParser()
ap.add_argument("--src", nargs="+", required=True)
ap.add_argument("--dst", required=True)
ap.add_argument("--weights", nargs="+", type=float, default=None)
a = ap.parse_args()

w = a.weights or [1.0 / len(a.src)] * len(a.src)
assert len(w) == len(a.src)
print("averaging", list(zip(a.src, w)), flush=True)

model = Gemma3ForConditionalGeneration.from_pretrained(a.src[0], dtype=torch.float32)
sd = model.state_dict()
for k in sd:
    sd[k].mul_(w[0])

for path, wi in zip(a.src[1:], w[1:]):
    other = Gemma3ForConditionalGeneration.from_pretrained(path, dtype=torch.float32)
    osd = other.state_dict()
    missing = set(sd) ^ set(osd)
    assert not missing, f"key mismatch: {sorted(missing)[:5]}"
    for k in sd:
        sd[k].add_(osd[k], alpha=wi)
    del other, osd

model.load_state_dict(sd)
model = model.to(torch.bfloat16)
os.makedirs(a.dst, exist_ok=True)
model.save_pretrained(a.dst)
AutoTokenizer.from_pretrained(a.src[0]).save_pretrained(a.dst)
for fn in ("preprocessor_config.json", "processor_config.json"):
    s = os.path.join(a.src[0], fn)
    if os.path.exists(s):
        shutil.copy(s, os.path.join(a.dst, fn))

p = os.path.join(a.dst, "generation_config.json")
g = json.load(open(p))
g["do_sample"] = False
g["temperature"] = 0.0
g.pop("top_k", None)
g.pop("top_p", None)
json.dump(g, open(p, "w"), indent=2)
print("wrote", a.dst, json.dumps(g), flush=True)
