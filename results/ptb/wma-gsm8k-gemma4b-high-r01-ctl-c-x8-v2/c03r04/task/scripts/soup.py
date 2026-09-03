#!/usr/bin/env python3
"""Uniform weight average of two checkpoints that lie on the same training trajectory."""
from __future__ import annotations

import argparse
import json
import os
import shutil

import torch
from safetensors.torch import load_file, save_file


def shard_map(d):
    idx = json.load(open(os.path.join(d, "model.safetensors.index.json")))
    return idx, sorted(set(idx["weight_map"].values()))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--alpha", type=float, default=0.5, help="weight on --a")
    ap.add_argument("--tokenizer-from", required=True)
    ap.add_argument("--generation-config", required=True)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    ia, sa = shard_map(args.a)
    ib, sb = shard_map(args.b)
    assert sa == sb, (sa, sb)
    assert ia["weight_map"] == ib["weight_map"], "shard layouts differ"

    n = 0
    for sh in sa:
        ta = load_file(os.path.join(args.a, sh))
        tb = load_file(os.path.join(args.b, sh))
        assert set(ta) == set(tb)
        out = {}
        for k in ta:
            x, y = ta[k], tb[k]
            if x.is_floating_point():
                out[k] = (args.alpha * x.float() + (1 - args.alpha) * y.float()).to(x.dtype)
            else:
                assert torch.equal(x, y), k
                out[k] = x
            n += 1
        save_file(out, os.path.join(args.out, sh), metadata={"format": "pt"})
        del ta, tb, out
        print("wrote", sh, flush=True)
    shutil.copy(os.path.join(args.a, "model.safetensors.index.json"), args.out)
    shutil.copy(os.path.join(args.a, "config.json"), args.out)
    for fn in ("added_tokens.json", "special_tokens_map.json", "tokenizer.json",
               "tokenizer.model", "tokenizer_config.json", "preprocessor_config.json",
               "processor_config.json"):
        p = os.path.join(args.tokenizer_from, fn)
        if os.path.exists(p):
            shutil.copy(os.path.realpath(p), args.out)
    shutil.copy(os.path.realpath(args.generation_config),
                os.path.join(args.out, "generation_config.json"))
    print(f"averaged {n} tensors (alpha={args.alpha}) -> {args.out}")


if __name__ == "__main__":
    main()
