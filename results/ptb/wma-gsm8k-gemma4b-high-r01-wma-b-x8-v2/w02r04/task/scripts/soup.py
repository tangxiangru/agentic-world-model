#!/usr/bin/env python3
"""Uniform weight average of two or more checkpoints of the same architecture."""
from __future__ import annotations

import argparse
import glob
import json
import os
import shutil

import torch
from safetensors.torch import load_file, save_file


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    ref = args.ckpts[0]
    for f in os.listdir(ref):
        if not f.endswith(".safetensors"):
            shutil.copyfile(os.path.join(ref, f), os.path.join(args.out, f))

    shards = sorted(os.path.basename(p) for p in glob.glob(os.path.join(ref, "*.safetensors")))
    n = len(args.ckpts)
    for shard in shards:
        acc = None
        for c in args.ckpts:
            sd = load_file(os.path.join(c, shard))
            if acc is None:
                acc = {k: v.to(torch.float32) for k, v in sd.items()}
            else:
                for k in acc:
                    acc[k] += sd[k].to(torch.float32)
            del sd
        out = {k: (v / n).to(torch.bfloat16) for k, v in acc.items()}
        save_file(out, os.path.join(args.out, shard), metadata={"format": "pt"})
        print("wrote", shard, len(out), "tensors", flush=True)
    print("souped", n, "checkpoints ->", args.out)


if __name__ == "__main__":
    main()
