#!/usr/bin/env python3
"""Uniform weight average of two checkpoints of the same lineage."""
import argparse
import json
import os
import shutil

import torch
from safetensors.torch import load_file, save_file

ap = argparse.ArgumentParser()
ap.add_argument("--a", default="ckpt/sft1")
ap.add_argument("--b", default="ckpt/sft2")
ap.add_argument("--alpha", type=float, default=0.5, help="weight on --b")
ap.add_argument("--out", default="ckpt/soup")
args = ap.parse_args()

os.makedirs(args.out, exist_ok=True)
index = json.load(open(os.path.join(args.b, "model.safetensors.index.json")))
shards = sorted(set(index["weight_map"].values()))
for sh in shards:
    ta = load_file(os.path.join(args.a, sh))
    tb = load_file(os.path.join(args.b, sh))
    merged = {}
    for k, v in tb.items():
        if k in ta and v.is_floating_point():
            merged[k] = (ta[k].to(torch.float32) * (1 - args.alpha)
                         + v.to(torch.float32) * args.alpha).to(v.dtype)
        else:
            merged[k] = v
    save_file(merged, os.path.join(args.out, sh), metadata={"format": "pt"})
    print("merged", sh)

for fn in os.listdir(args.b):
    if fn.endswith(".safetensors"):
        continue
    shutil.copy(os.path.join(args.b, fn), os.path.join(args.out, fn))
print("wrote", args.out)
