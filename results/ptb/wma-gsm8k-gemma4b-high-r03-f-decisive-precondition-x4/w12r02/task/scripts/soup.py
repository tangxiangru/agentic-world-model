#!/usr/bin/env python3
"""Uniform (or weighted) parameter average of several checkpoints of the same model."""
from __future__ import annotations

import argparse
import json
import os
import shutil

import torch
from safetensors.torch import load_file, save_file


def shard_files(path):
    idx = os.path.join(path, "model.safetensors.index.json")
    if os.path.exists(idx):
        wm = json.load(open(idx))["weight_map"]
        return sorted(set(wm.values()))
    return ["model.safetensors"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--weights", nargs="*", type=float, default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    w = args.weights or [1.0 / len(args.inputs)] * len(args.inputs)
    assert len(w) == len(args.inputs)
    s = sum(w)
    w = [x / s for x in w]
    print("weights:", dict(zip(args.inputs, w)))

    base = args.inputs[0]
    os.makedirs(args.out, exist_ok=True)
    for fn in os.listdir(base):
        if fn.endswith(".safetensors"):
            continue
        shutil.copy(os.path.join(base, fn), os.path.join(args.out, fn))

    for sf in shard_files(base):
        acc = None
        for p, wi in zip(args.inputs, w):
            t = load_file(os.path.join(p, sf))
            if acc is None:
                acc = {k: v.to(torch.float32) * wi for k, v in t.items()}
            else:
                for k in acc:
                    acc[k] += t[k].to(torch.float32) * wi
            del t
        save_file({k: v.to(torch.bfloat16) for k, v in acc.items()},
                  os.path.join(args.out, sf), metadata={"format": "pt"})
        print("wrote", sf, flush=True)
        del acc
    print("soup ->", args.out)


if __name__ == "__main__":
    main()
