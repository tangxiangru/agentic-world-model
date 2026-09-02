#!/usr/bin/env python3
"""Uniform weight average of two checkpoints of the same architecture."""
from __future__ import annotations

import argparse
import json
import os
import shutil

import torch
from safetensors.torch import load_file, save_file


def shard_files(d: str) -> list[str]:
    idx = os.path.join(d, "model.safetensors.index.json")
    if os.path.exists(idx):
        m = json.load(open(idx))["weight_map"]
        return sorted(set(m.values()))
    return ["model.safetensors"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--weight-a", type=float, default=0.5)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    files = shard_files(args.a)
    assert files == shard_files(args.b), "shard layouts differ"

    for fn in files:
        ta = load_file(os.path.join(os.path.realpath(args.a), fn))
        tb = load_file(os.path.join(os.path.realpath(args.b), fn))
        assert set(ta) == set(tb), f"key mismatch in {fn}"
        out = {}
        for k in ta:
            x, y = ta[k], tb[k]
            if x.is_floating_point():
                out[k] = (
                    args.weight_a * x.to(torch.float32) + (1 - args.weight_a) * y.to(torch.float32)
                ).to(x.dtype)
            else:
                assert torch.equal(x, y), k
                out[k] = x
        save_file(out, os.path.join(args.out, fn), metadata={"format": "pt"})
        print("wrote", fn, flush=True)

    for fn in os.listdir(args.a):
        if fn in files or os.path.isdir(os.path.join(args.a, fn)):
            continue
        shutil.copy2(os.path.realpath(os.path.join(args.a, fn)), os.path.join(args.out, fn))
    print("soup at", args.out)


if __name__ == "__main__":
    main()
