#!/usr/bin/env python3
"""Uniform weight average ('soup') of two or more checkpoints of the same shape.

Averaging is done shard-by-shard in fp32 and written back in the source dtype;
the tokenizer / config / processor files are taken from the first checkpoint.
"""
import argparse
import json
import os
import shutil

import torch
from safetensors.torch import load_file, save_file


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("ckpts", nargs="+")
    a = ap.parse_args()

    src = a.ckpts[0]
    os.makedirs(a.out, exist_ok=True)
    for f in os.listdir(src):
        if f.endswith(".safetensors") or f == "training_args.bin":
            continue
        shutil.copyfile(os.path.join(src, f), os.path.join(a.out, f))

    index = json.load(open(os.path.join(src, "model.safetensors.index.json")))
    shards = sorted(set(index["weight_map"].values()))
    for shard in shards:
        acc = None
        for c in a.ckpts:
            sd = load_file(os.path.join(c, shard))
            if acc is None:
                acc = {k: v.to(torch.float32) for k, v in sd.items()}
            else:
                for k in acc:
                    acc[k] += sd[k].to(torch.float32)
            del sd
        ref = load_file(os.path.join(src, shard))
        out = {k: (acc[k] / len(a.ckpts)).to(ref[k].dtype) for k in acc}
        save_file(out, os.path.join(a.out, shard), metadata={"format": "pt"})
        print("wrote", shard)
        del acc, ref, out
    print("soup of", len(a.ckpts), "->", a.out)


if __name__ == "__main__":
    main()
