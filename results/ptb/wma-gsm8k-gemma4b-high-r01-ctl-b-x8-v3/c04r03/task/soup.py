#!/usr/bin/env python3
"""Uniform weight average of two or more checkpoints of the same architecture.

Loads shard by shard on CPU so the 4B model never needs more than one copy in
RAM per input. Copies tokenizer/processor/generation config from the first
input so the result stays a directory the grader can load unchanged.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--weights", nargs="*", type=float, default=None)
    args = ap.parse_args()

    ws = args.weights or [1.0 / len(args.inputs)] * len(args.inputs)
    assert len(ws) == len(args.inputs)
    s = sum(ws)
    ws = [w / s for w in ws]
    print("weights:", dict(zip(args.inputs, ws)))

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    first = Path(args.inputs[0])
    for f in first.iterdir():
        if f.suffix == ".safetensors":
            continue
        shutil.copy(f, out / f.name)

    shards = sorted(p.name for p in first.glob("*.safetensors"))
    for sh in shards:
        acc = None
        for w, d in zip(ws, args.inputs):
            sd = load_file(str(Path(d) / sh))
            if acc is None:
                acc = {k: v.to(torch.float32) * w for k, v in sd.items()}
            else:
                for k in acc:
                    acc[k] += sd[k].to(torch.float32) * w
            del sd
        acc = {k: v.to(torch.bfloat16) for k, v in acc.items()}
        save_file(acc, str(out / sh), metadata={"format": "pt"})
        print("wrote", sh, flush=True)
    print("soup at", out)


if __name__ == "__main__":
    main()
