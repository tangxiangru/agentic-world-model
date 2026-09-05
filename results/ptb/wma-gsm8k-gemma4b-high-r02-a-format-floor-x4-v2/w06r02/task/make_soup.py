#!/usr/bin/env python3
"""Uniformly average the weights of two checkpoints from the same lineage and save the
result as a loadable model dir (weights only; configs are copied from the first source)."""
import argparse
import json
import os
import shutil

import torch
from safetensors.torch import load_file, save_file

ap = argparse.ArgumentParser()
ap.add_argument("--a", required=True)
ap.add_argument("--b", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--weight-a", type=float, default=0.5)
args = ap.parse_args()

os.makedirs(args.out, exist_ok=True)
index = json.load(open(os.path.join(args.a, "model.safetensors.index.json")))
shards = sorted(set(index["weight_map"].values()))
wa, wb = args.weight_a, 1.0 - args.weight_a
for shard in shards:
    ta = load_file(os.path.join(args.a, shard))
    tb = load_file(os.path.join(args.b, shard))
    assert ta.keys() == tb.keys(), shard
    out = {}
    for k, v in ta.items():
        out[k] = (v.to(torch.float32) * wa + tb[k].to(torch.float32) * wb).to(v.dtype)
    save_file(out, os.path.join(args.out, shard), metadata={"format": "pt"})
    print("merged", shard, len(out), "tensors", flush=True)

for f in os.listdir(args.a):
    if f.endswith(".safetensors") or f == "training_args.bin":
        continue
    src = os.path.realpath(os.path.join(args.a, f))
    if os.path.isfile(src):
        shutil.copy(src, os.path.join(args.out, f))
print("wrote", args.out)
