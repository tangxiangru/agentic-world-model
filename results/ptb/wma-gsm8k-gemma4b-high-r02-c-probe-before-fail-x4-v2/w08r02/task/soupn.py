#!/usr/bin/env python3
"""Weighted average of N checkpoints of the same architecture (float32 accumulate)."""
import argparse, json, os, shutil
import torch
from safetensors.torch import load_file, save_file

ap = argparse.ArgumentParser()
ap.add_argument("--models", nargs="+", required=True)
ap.add_argument("--weights", nargs="+", type=float, default=None)
ap.add_argument("--out", required=True)
a = ap.parse_args()
w = a.weights or [1.0] * len(a.models)
assert len(w) == len(a.models)
tot = sum(w); w = [x / tot for x in w]
print("weights:", dict(zip(a.models, w)))

os.makedirs(a.out, exist_ok=True)
idx = json.load(open(os.path.join(a.models[0], "model.safetensors.index.json")))
for sh in sorted(set(idx["weight_map"].values())):
    acc = None
    for m, wi in zip(a.models, w):
        t = load_file(os.path.join(m, sh))
        if acc is None:
            acc = {k: v.to(torch.float32) * wi for k, v in t.items()}
            dt = {k: v.dtype for k, v in t.items()}
        else:
            for k in acc:
                acc[k] += t[k].to(torch.float32) * wi
        del t
    save_file({k: v.to(dt[k]) for k, v in acc.items()}, os.path.join(a.out, sh), metadata={"format": "pt"})
    print("wrote", sh, flush=True)
    del acc
for f in ["model.safetensors.index.json", "config.json"]:
    shutil.copy2(os.path.join(a.models[0], f), os.path.join(a.out, f))
print("soup at", a.out)
