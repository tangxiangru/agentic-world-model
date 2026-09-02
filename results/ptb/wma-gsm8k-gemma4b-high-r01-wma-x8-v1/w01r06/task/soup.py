#!/usr/bin/env python3
"""Uniform weight average of two or more checkpoints of the same architecture.

All inputs descend from the same base snapshot by successive fine-tuning, so
they sit in one basin and averaging is well defined. Tokenizer and config are
copied from the first source.
"""
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
    ap.add_argument("--src", nargs="+", required=True)
    ap.add_argument("--dst", required=True)
    args = ap.parse_args()

    if os.path.exists(args.dst):
        shutil.rmtree(args.dst)
    os.makedirs(args.dst)
    for f in os.listdir(args.src[0]):
        if not f.endswith(".safetensors"):
            shutil.copy(os.path.join(args.src[0], f), os.path.join(args.dst, f))

    shards = sorted(os.path.basename(p) for p in
                    glob.glob(os.path.join(args.src[0], "*.safetensors")))
    n = len(args.src)
    for shard in shards:
        acc = None
        for src in args.src:
            sd = load_file(os.path.join(src, shard))
            if acc is None:
                acc = {k: v.to(torch.float32) for k, v in sd.items()}
            else:
                assert acc.keys() == sd.keys(), shard
                for k in acc:
                    acc[k] += sd[k].to(torch.float32)
            del sd
        out = {k: (v / n).to(torch.bfloat16) for k, v in acc.items()}
        save_file(out, os.path.join(args.dst, shard),
                  metadata={"format": "pt"})
        print("wrote", shard, len(out), "tensors", flush=True)
        del acc, out

    print(json.dumps({"sources": args.src, "dst": args.dst, "n": n}, indent=2))


if __name__ == "__main__":
    main()
