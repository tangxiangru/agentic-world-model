#!/usr/bin/env python3
"""Uniform weight average ("model soup") of two or more checkpoints of the same base.

Averages shard by shard so peak RAM stays at a couple of shards, and copies the
config/tokenizer side-cars from the first input.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    srcs = [Path(p) for p in args.inputs]
    dst = Path(args.out)
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True)

    shards = sorted(p.name for p in srcs[0].glob("*.safetensors"))
    for s in srcs[1:]:
        assert sorted(p.name for p in s.glob("*.safetensors")) == shards, f"shard mismatch in {s}"

    for name in shards:
        acc = None
        for s in srcs:
            sd = load_file(str(s / name))
            if acc is None:
                acc = {k: v.to(torch.float32) for k, v in sd.items()}
            else:
                for k in acc:
                    acc[k] += sd[k].to(torch.float32)
            del sd
        n = len(srcs)
        out = {k: (v / n).to(torch.float32) for k, v in acc.items()}
        save_file(out, str(dst / name), metadata={"format": "pt"})
        print(f"averaged {name} over {n} checkpoints")
        del acc, out

    for p in srcs[0].iterdir():
        if p.suffix == ".safetensors" or p.name == "training_args.bin":
            continue
        shutil.copy2(p, dst / p.name)
    print(f"wrote {dst}")


if __name__ == "__main__":
    main()
