#!/usr/bin/env python3
"""Uniform weight average ("model soup") of checkpoints that share an initialisation.

Streams the shards so peak RAM stays at one tensor per parameter, and copies the
tokenizer/config files from the first source so the result loads like any other
checkpoint.
"""
import argparse, json, os, shutil
import torch
from safetensors.torch import load_file, save_file

ap = argparse.ArgumentParser()
ap.add_argument("--srcs", nargs="+", required=True)
ap.add_argument("--dst", required=True)
ap.add_argument("--weights", nargs="*", type=float, default=None)
a = ap.parse_args()

w = a.weights or [1.0 / len(a.srcs)] * len(a.srcs)
assert len(w) == len(a.srcs)
w = [x / sum(w) for x in w]
print("weights:", dict(zip(a.srcs, w)))

os.makedirs(a.dst, exist_ok=True)
idx = json.load(open(os.path.join(a.srcs[0], "model.safetensors.index.json")))
shards = sorted(set(idx["weight_map"].values()))

for shard in shards:
    acc = None
    for src, wi in zip(a.srcs, w):
        sd = load_file(os.path.join(src, shard))
        if acc is None:
            acc = {k: v.to(torch.float32) * wi for k, v in sd.items()}
        else:
            assert set(sd) == set(acc), f"{src}/{shard} has a different key set"
            for k in acc:
                acc[k] += sd[k].to(torch.float32) * wi
        del sd
    out = {k: v.to(torch.bfloat16) for k, v in acc.items()}
    save_file(out, os.path.join(a.dst, shard), metadata={"format": "pt"})
    print("wrote", shard, len(out), "tensors", flush=True)
    del acc, out

for fn in os.listdir(a.srcs[0]):
    if fn.endswith(".safetensors") or fn == "training_args.bin":
        continue
    s = os.path.join(a.srcs[0], fn)
    if os.path.isfile(s):
        shutil.copy2(s, os.path.join(a.dst, fn))
print("soup at", a.dst)
