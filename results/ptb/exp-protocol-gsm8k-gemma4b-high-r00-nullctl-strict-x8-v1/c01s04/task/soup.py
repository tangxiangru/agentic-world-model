#!/usr/bin/env python3
"""Uniformly average the weights of several checkpoints."""
import argparse
import json
import os
import shutil

import torch
from safetensors.torch import load_file, save_file

ap = argparse.ArgumentParser()
ap.add_argument("--srcs", nargs="+", required=True)
ap.add_argument("--dst", required=True)
a = ap.parse_args()

os.makedirs(a.dst, exist_ok=True)
idx = json.load(open(os.path.join(a.srcs[0], "model.safetensors.index.json")))
for fn in sorted(set(idx["weight_map"].values())):
    acc = None
    for s in a.srcs:
        sd = load_file(os.path.join(s, fn))
        if acc is None:
            acc = {k: v.to(torch.float32) for k, v in sd.items()}
        else:
            for k in acc:
                acc[k] += sd[k].to(torch.float32)
        del sd
    save_file({k: (v / len(a.srcs)).to(torch.bfloat16) for k, v in acc.items()},
              os.path.join(a.dst, fn), metadata={"format": "pt"})
    print("wrote", fn, flush=True)
    del acc
for f in os.listdir(a.srcs[-1]):
    if f.endswith(".safetensors") or f in ("optimizer.pt", "rng_state.pth", "scheduler.pt",
                                           "trainer_state.json", "training_args.bin"):
        continue
    p = os.path.join(a.srcs[-1], f)
    if os.path.isfile(p):
        shutil.copy(p, os.path.join(a.dst, f))
print("soup ->", a.dst)
