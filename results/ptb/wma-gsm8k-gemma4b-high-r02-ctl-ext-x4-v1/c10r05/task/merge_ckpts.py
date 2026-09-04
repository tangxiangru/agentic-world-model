#!/usr/bin/env python3
"""Uniform weight average (model soup) of two or more checkpoints of the same architecture."""
from __future__ import annotations

import argparse
import json
import os
import shutil

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE = "/home/ben/hf_cache/hub/models--google--gemma-3-4b-pt/snapshots/cc012e0a6d0787b4adcc0fa2c4da74402494554d"

ap = argparse.ArgumentParser()
ap.add_argument("--srcs", nargs="+", required=True)
ap.add_argument("--weights", nargs="*", type=float, default=None)
ap.add_argument("--out", required=True)
a = ap.parse_args()

w = a.weights or [1.0 / len(a.srcs)] * len(a.srcs)
assert len(w) == len(a.srcs)
w = [x / sum(w) for x in w]
print("soup:", list(zip(a.srcs, w)), flush=True)

model = AutoModelForCausalLM.from_pretrained(a.srcs[0], dtype=torch.float32)
sd = model.state_dict()
for k in sd:
    sd[k].mul_(w[0])
for src, wi in zip(a.srcs[1:], w[1:]):
    other = AutoModelForCausalLM.from_pretrained(src, dtype=torch.float32)
    osd = other.state_dict()
    for k in sd:
        sd[k].add_(osd[k], alpha=wi)
    del other, osd
model.load_state_dict(sd)
model = model.to(torch.bfloat16)

g = model.generation_config
g.do_sample = False
for k in ("temperature", "top_p", "top_k"):
    if getattr(g, k, None) is not None:
        setattr(g, k, None)
g.validate()
model.save_pretrained(a.out)
AutoTokenizer.from_pretrained(BASE).save_pretrained(a.out)
for fn in ("preprocessor_config.json", "processor_config.json"):
    shutil.copy(os.path.join(BASE, fn), os.path.join(a.out, fn))
json.dump(
    {
        "bos_token_id": 2,
        "eos_token_id": [1, 106],
        "pad_token_id": 0,
        "do_sample": False,
        "temperature": 0.0,
        "cache_implementation": "hybrid",
        "transformers_version": "4.57.3",
    },
    open(os.path.join(a.out, "generation_config.json"), "w"),
    indent=2,
)
print("saved", a.out, sorted(os.listdir(a.out)))
