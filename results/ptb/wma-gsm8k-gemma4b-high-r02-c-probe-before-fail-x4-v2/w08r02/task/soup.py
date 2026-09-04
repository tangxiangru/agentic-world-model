#!/usr/bin/env python3
"""Uniform weight average of two checkpoints of the same architecture.

exp-04 is a continuation of exp-03, so the two sit in the same loss basin and a
uniform soup is well defined. Averaging is done in float32 and cast back to the
source dtype.
"""
import argparse, json, os, shutil, glob
import torch
from safetensors.torch import load_file, save_file

ap = argparse.ArgumentParser()
ap.add_argument("--a", required=True)
ap.add_argument("--b", required=True)
ap.add_argument("--out", required=True)
args = ap.parse_args()

os.makedirs(args.out, exist_ok=True)
idx = json.load(open(os.path.join(args.a, "model.safetensors.index.json")))
shards = sorted(set(idx["weight_map"].values()))
for sh in shards:
    ta = load_file(os.path.join(args.a, sh))
    tb = load_file(os.path.join(args.b, sh))
    assert set(ta) == set(tb), sh
    out = {}
    for k, v in ta.items():
        out[k] = ((v.to(torch.float32) + tb[k].to(torch.float32)) / 2).to(v.dtype)
    save_file(out, os.path.join(args.out, sh), metadata={"format": "pt"})
    print("wrote", sh, flush=True)
    del ta, tb, out
for f in ["model.safetensors.index.json", "config.json"]:
    shutil.copy2(os.path.join(args.a, f), os.path.join(args.out, f))
print("soup at", args.out)
