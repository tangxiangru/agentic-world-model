#!/usr/bin/env python3
"""Uniform weight average of two checkpoints with identical parameter shapes."""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--out", required=True)
    ap.add_argument("--w", type=float, default=0.5, help="weight on a")
    args = ap.parse_args()

    a, b, out = Path(args.a), Path(args.b), Path(args.out)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    for f in a.iterdir():
        if f.is_file() and not f.name.endswith(".safetensors"):
            shutil.copy2(f, out / f.name)
    shards = sorted(p.name for p in a.iterdir() if p.name.endswith(".safetensors"))
    for name in shards:
        ta, tb = load_file(a / name), load_file(b / name)
        assert set(ta) == set(tb), name
        merged = {}
        for k in ta:
            x, y = ta[k], tb[k]
            merged[k] = (args.w * x.float() + (1 - args.w) * y.float()).to(x.dtype)
        save_file(merged, str(out / name), metadata={"format": "pt"})
        print("merged", name, flush=True)
    print("wrote", out)


if __name__ == "__main__":
    main()
