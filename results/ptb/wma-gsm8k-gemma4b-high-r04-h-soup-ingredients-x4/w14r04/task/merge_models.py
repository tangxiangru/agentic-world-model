#!/usr/bin/env python3
"""Uniform weight-space average (model soup) of two or more checkpoints.

Only valid for checkpoints in the same basin - here every arm is a sequential
continuation of the same SFT run, so they are. Runs on CPU in bf16 with fp32
accumulation, then writes a directory finalize_model.py can turn into a
scoreable model.
"""
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
    ap.add_argument("--srcs", nargs="+", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--weights", nargs="*", type=float, default=None)
    args = ap.parse_args()

    w = args.weights or [1.0 / len(args.srcs)] * len(args.srcs)
    assert len(w) == len(args.srcs)
    s = sum(w)
    w = [x / s for x in w]
    print("weights:", dict(zip(args.srcs, w)))

    os.makedirs(args.dst, exist_ok=True)
    files = shard_files(args.srcs[0])
    for f in files:
        acc = None
        for src, wi in zip(args.srcs, w):
            sd = load_file(os.path.join(src, f))
            if acc is None:
                acc = {k: v.to(torch.float32) * wi for k, v in sd.items()}
            else:
                assert set(sd) == set(acc), f"tensor sets differ in {f}"
                for k in acc:
                    acc[k] += sd[k].to(torch.float32) * wi
            del sd
        out = {k: v.to(torch.bfloat16) for k, v in acc.items()}
        save_file(out, os.path.join(args.dst, f), metadata={"format": "pt"})
        print("wrote", f, len(out), "tensors")
        del acc, out

    for extra in ("config.json", "model.safetensors.index.json", "generation_config.json"):
        p = os.path.join(args.srcs[0], extra)
        if os.path.exists(p):
            shutil.copy2(p, os.path.join(args.dst, extra))
    print("soup written to", args.dst)


if __name__ == "__main__":
    main()
