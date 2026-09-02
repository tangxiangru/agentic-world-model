#!/usr/bin/env python3
"""Uniform weight average of two checkpoints on the same training trajectory.

Writes a complete, self-contained model directory (real files, tokenizer,
processor, greedy generation_config) so the grader can load it directly.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil

import torch
from safetensors.torch import load_file, save_file


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--alpha", type=float, default=0.5, help="weight on --a")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    idx = json.load(open(os.path.join(args.a, "model.safetensors.index.json")))
    shards = sorted(set(idx["weight_map"].values()))

    for s in shards:
        ta = load_file(os.path.join(args.a, s))
        tb = load_file(os.path.join(args.b, s))
        assert set(ta) == set(tb), f"key mismatch in {s}"
        merged = {}
        for k, va in ta.items():
            vb = tb[k]
            if va.dtype.is_floating_point:
                merged[k] = (args.alpha * va.float() + (1 - args.alpha) * vb.float()).to(va.dtype)
            else:
                assert torch.equal(va, vb), f"non-float tensor differs: {k}"
                merged[k] = va
        save_file(merged, os.path.join(args.out, s), metadata={"format": "pt"})
        print("merged", s, flush=True)
        del ta, tb, merged

    for name in os.listdir(args.a):
        if name.endswith(".safetensors") or name == "training_args.bin":
            continue
        src = os.path.join(args.a, name)
        if os.path.isfile(src):
            shutil.copyfile(src, os.path.join(args.out, name))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
