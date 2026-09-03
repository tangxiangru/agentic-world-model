#!/usr/bin/env python3
"""Uniform weight average (model soup) of checkpoints fine-tuned from the same base."""
import argparse, json, os, shutil, sys
import torch
from safetensors.torch import load_file, save_file

ap = argparse.ArgumentParser()
ap.add_argument("--inputs", nargs="+", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()

os.makedirs(a.out, exist_ok=True)
idx = json.load(open(os.path.join(a.inputs[0], "model.safetensors.index.json")))
shards = sorted(set(idx["weight_map"].values()))
for shard in shards:
    acc = None
    for d in a.inputs:
        sd = load_file(os.path.join(d, shard))
        if acc is None:
            acc = {k: v.to(torch.float32) for k, v in sd.items()}
        else:
            for k in acc:
                acc[k] += sd[k].to(torch.float32)
        del sd
    n = len(a.inputs)
    out = {k: (v / n).to(torch.bfloat16) for k, v in acc.items()}
    save_file(out, os.path.join(a.out, shard), metadata={"format": "pt"})
    print("wrote", shard, flush=True)
    del acc, out
for f in ("model.safetensors.index.json", "config.json"):
    shutil.copy(os.path.join(a.inputs[0], f), os.path.join(a.out, f))
print("soup at", a.out)
