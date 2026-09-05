#!/usr/bin/env python3
"""Linear interpolation of two checkpoints that lie on the same fine-tuning path.

out = (1 - alpha) * A + alpha * B, tensor by tensor, in the dtype A stores.
Everything that is not a weight (tokenizer, configs) is taken from B.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil

import torch
from safetensors.torch import load_file, save_file


def shard_map(path):
    idx = os.path.join(path, "model.safetensors.index.json")
    with open(idx) as f:
        return json.load(f)["weight_map"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    for fn in os.listdir(args.b):
        if fn.endswith(".safetensors"):
            continue
        src = os.path.join(args.b, fn)
        if os.path.isfile(src):
            shutil.copy(src, os.path.join(args.out, fn))

    wm_a, wm_b = shard_map(args.a), shard_map(args.b)
    assert set(wm_a) == set(wm_b), "checkpoints have different parameter sets"
    shards = sorted(set(wm_b.values()))
    for shard in shards:
        keys = [k for k, v in wm_b.items() if v == shard]
        ta = {}
        for k in keys:
            sa = wm_a[k]
            ta.setdefault(sa, None)
        cache_a = {s: load_file(os.path.join(args.a, s)) for s in ta}
        tb = load_file(os.path.join(args.b, shard))
        out = {}
        for k in keys:
            a = cache_a[wm_a[k]][k]
            b = tb[k]
            dt = b.dtype
            out[k] = ((1 - args.alpha) * a.float() + args.alpha * b.float()).to(dt)
        save_file(out, os.path.join(args.out, shard), metadata={"format": "pt"})
        print(f"wrote {shard} ({len(keys)} tensors)", flush=True)
        del cache_a, tb, out
        torch.cuda.empty_cache()
    print("done ->", args.out, flush=True)


if __name__ == "__main__":
    main()
