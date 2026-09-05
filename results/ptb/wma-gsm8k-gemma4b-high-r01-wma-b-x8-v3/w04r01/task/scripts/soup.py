"""Uniform weight average of two checkpoints of the same architecture."""
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
    for sh in shards:
        ta = load_file(os.path.join(args.a, sh))
        tb = load_file(os.path.join(args.b, sh))
        assert set(ta) == set(tb), sh
        merged = {}
        for k, v in ta.items():
            merged[k] = (
                v.to(torch.float32) * args.alpha + tb[k].to(torch.float32) * (1 - args.alpha)
            ).to(v.dtype)
        save_file(merged, os.path.join(args.out, sh), metadata={"format": "pt"})
        del ta, tb, merged
        print("merged", sh, flush=True)
    for f in ("model.safetensors.index.json", "config.json"):
        shutil.copy2(os.path.join(args.a, f), os.path.join(args.out, f))
    print("SOUP", args.out)


if __name__ == "__main__":
    main()
