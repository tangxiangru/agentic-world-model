"""Weight-average two checkpoints from the same training trajectory (model soup).

exp-04/final is a continuation of exp-02/final, so the two sit on one trajectory
and their average is a cheap approximation of a longer-horizon weight average.
Costs no training; the output is a normal checkpoint directory.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil

import torch
from safetensors.torch import load_file, save_file

SNAP = "/home/ben/hf_cache/hub/models--google--gemma-3-4b-pt/snapshots/cc012e0a6d0787b4adcc0fa2c4da74402494554d"
SIDECARS = ["config.json", "tokenizer.json", "tokenizer.model", "tokenizer_config.json",
            "special_tokens_map.json", "added_tokens.json", "preprocessor_config.json",
            "processor_config.json", "generation_config.json", "model.safetensors.index.json"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--weight-b", type=float, default=0.5)
    ap.add_argument("--dst", required=True)
    args = ap.parse_args()

    os.makedirs(args.dst, exist_ok=True)
    for f in SIDECARS:
        for src in (args.a, SNAP):
            p = os.path.join(src, f)
            if os.path.exists(p):
                shutil.copy(p, os.path.join(args.dst, f))
                break

    index = json.load(open(os.path.join(args.a, "model.safetensors.index.json")))
    shards = sorted(set(index["weight_map"].values()))
    wb = args.weight_b
    n = 0
    for shard in shards:
        sa = load_file(os.path.join(args.a, shard))
        sb = load_file(os.path.join(args.b, shard))
        assert sa.keys() == sb.keys(), shard
        out = {}
        for k in sa:
            ta, tb = sa[k], sb[k]
            if ta.is_floating_point():
                out[k] = ((1 - wb) * ta.to(torch.float32) + wb * tb.to(torch.float32)).to(ta.dtype)
                n += 1
            else:
                out[k] = ta
        save_file(out, os.path.join(args.dst, shard), metadata={"format": "pt"})
        print("wrote", shard, len(out), "tensors")
    print(f"averaged {n} float tensors with weight_b={wb} -> {args.dst}")


if __name__ == "__main__":
    main()
