"""Uniform weight average ("model soup") of two full fine-tunes of the same base.

Both parents were trained from the same immutable snapshot with the same recipe,
so their parameters are index-aligned and averaging is well defined.
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
    with open(idx) as f:
        weight_map = json.load(f)["weight_map"]
    return sorted(set(weight_map.values()))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--alpha", type=float, default=0.5, help="weight on --a")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    shards_a, shards_b = shard_files(args.a), shard_files(args.b)
    assert shards_a == shards_b, (shards_a, shards_b)

    if os.path.exists(args.out):
        shutil.rmtree(args.out)
    os.makedirs(args.out)
    for fn in os.listdir(args.b):
        if not fn.endswith(".safetensors"):
            shutil.copy(os.path.join(args.b, fn), os.path.join(args.out, fn))

    n_params, n_tensors = 0, 0
    for shard in shards_a:
        ta = load_file(os.path.join(args.a, shard))
        tb = load_file(os.path.join(args.b, shard))
        assert set(ta) == set(tb), shard
        merged = {}
        for k in ta:
            assert ta[k].shape == tb[k].shape, k
            if ta[k].is_floating_point():
                merged[k] = (
                    args.alpha * ta[k].to(torch.float32)
                    + (1 - args.alpha) * tb[k].to(torch.float32)
                ).to(ta[k].dtype)
            else:
                merged[k] = tb[k].clone()
            n_params += ta[k].numel()
            n_tensors += 1
        save_file(merged, os.path.join(args.out, shard), metadata={"format": "pt"})
        del ta, tb, merged
        print(f"[soup] {shard} done", flush=True)

    print(json.dumps({"out": args.out, "alpha": args.alpha, "tensors": n_tensors,
                      "params": n_params}, indent=2))


if __name__ == "__main__":
    main()
