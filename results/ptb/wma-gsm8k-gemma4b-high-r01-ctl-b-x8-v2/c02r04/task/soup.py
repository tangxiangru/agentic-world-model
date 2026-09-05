#!/usr/bin/env python3
"""Uniform weight average of two checkpoints on the same trajectory.

exp-05 is a continuation of exp-04, so averaging them is a Polyak-style average
along one optimisation path rather than a merge of two independent models.
Runs on CPU, shard by shard, so it does not touch the GPU.
"""
from __future__ import annotations

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
ap.add_argument("--alpha", type=float, default=0.5, help="weight on --a")
args = ap.parse_args()

os.makedirs(args.out, exist_ok=True)
index = json.load(open(os.path.join(args.a, "model.safetensors.index.json")))
shards = sorted(set(index["weight_map"].values()))

for shard in shards:
    ta = load_file(os.path.join(args.a, shard))
    tb = load_file(os.path.join(args.b, shard))
    assert ta.keys() == tb.keys(), shard
    merged = {}
    for k in ta:
        if ta[k].is_floating_point():
            merged[k] = (args.alpha * ta[k].float()
                         + (1 - args.alpha) * tb[k].float()).to(ta[k].dtype)
        else:
            merged[k] = ta[k]
    save_file(merged, os.path.join(args.out, shard), metadata={"format": "pt"})
    print("merged", shard, len(merged), "tensors", flush=True)
    del ta, tb, merged

for fn in os.listdir(args.a):
    src = os.path.realpath(os.path.join(args.a, fn))
    if os.path.isfile(src) and not fn.endswith(".safetensors") and fn != "training_args.bin":
        shutil.copy(src, os.path.join(args.out, fn))
print("wrote", args.out)
