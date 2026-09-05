#!/usr/bin/env python3
"""Uniform weight average ("soup") of several checkpoints on the same
trajectory, written out as a normal HF checkpoint.

All inputs must share the architecture and the parameter names; the tokenizer,
processor and generation_config are taken from the first input.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil

import torch
from safetensors.torch import load_file, save_file

AUX = [
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "preprocessor_config.json",
    "processor_config.json",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--weights", nargs="*", type=float, default=None)
    args = ap.parse_args()

    n = len(args.ckpts)
    w = args.weights or [1.0 / n] * n
    assert len(w) == n, (w, n)
    s = sum(w)
    w = [x / s for x in w]
    print(f"[soup] {n} checkpoints, weights {w}")

    os.makedirs(args.out, exist_ok=True)
    index_path = os.path.join(args.ckpts[0], "model.safetensors.index.json")
    index = json.load(open(index_path))
    shards = sorted(set(index["weight_map"].values()))

    for shard in shards:
        acc = None
        for ck, wi in zip(args.ckpts, w):
            sd = load_file(os.path.join(ck, shard))
            if acc is None:
                acc = {k: v.to(torch.float32) * wi for k, v in sd.items()}
            else:
                assert set(sd) == set(acc), f"key mismatch in {ck}/{shard}"
                for k in acc:
                    acc[k] += sd[k].to(torch.float32) * wi
            del sd
        out = {k: v.to(torch.bfloat16) for k, v in acc.items()}
        save_file(out, os.path.join(args.out, shard), metadata={"format": "pt"})
        print(f"[soup] wrote {shard} ({len(out)} tensors)", flush=True)
        del acc, out

    shutil.copy(index_path, os.path.join(args.out, "model.safetensors.index.json"))
    for f in AUX:
        src = os.path.join(args.ckpts[0], f)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(args.out, f))
    print(f"[soup] done -> {args.out}")


if __name__ == "__main__":
    main()
