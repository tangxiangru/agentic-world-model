#!/usr/bin/env python3
"""Uniformly average the weights of several checkpoints from the same lineage.

Runs on CPU in bf16 with a float32 accumulator, one tensor at a time, so it
never needs more than one shard's worth of memory per checkpoint.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil

import torch
from safetensors import safe_open
from safetensors.torch import save_file


def tensor_index(ckpt: str) -> dict[str, str]:
    idx = os.path.join(ckpt, "model.safetensors.index.json")
    if os.path.exists(idx):
        return json.load(open(idx))["weight_map"]
    return {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("ckpts", nargs="+")
    a = ap.parse_args()

    maps = [tensor_index(c) for c in a.ckpts]
    names = sorted(maps[0])
    for c, m in zip(a.ckpts[1:], maps[1:]):
        assert set(m) == set(names), f"{c} has a different tensor set"
    print(f"souping {len(a.ckpts)} checkpoints, {len(names)} tensors")

    os.makedirs(a.out, exist_ok=True)
    # group by the first checkpoint's shard layout so the output mirrors it
    shards: dict[str, list[str]] = {}
    for n in names:
        shards.setdefault(maps[0][n], []).append(n)

    handles = [
        {f: safe_open(os.path.join(c, f), framework="pt", device="cpu")
         for f in set(m.values())}
        for c, m in zip(a.ckpts, maps)
    ]

    weight_map = {}
    for shard, tnames in shards.items():
        out = {}
        for n in tnames:
            acc = None
            for h, m in zip(handles, maps):
                t = h[m[n]].get_tensor(n)
                acc = t.to(torch.float32) if acc is None else acc + t.to(torch.float32)
            out[n] = (acc / len(a.ckpts)).to(torch.bfloat16)
            weight_map[n] = shard
        save_file(out, os.path.join(a.out, shard), metadata={"format": "pt"})
        print("wrote", shard, len(out), "tensors")
        del out

    total = sum(os.path.getsize(os.path.join(a.out, s)) for s in shards)
    json.dump({"metadata": {"total_size": total}, "weight_map": weight_map},
              open(os.path.join(a.out, "model.safetensors.index.json"), "w"), indent=2)

    for f in ("config.json", "tokenizer.json", "tokenizer.model", "tokenizer_config.json",
              "special_tokens_map.json", "added_tokens.json", "preprocessor_config.json",
              "processor_config.json", "generation_config.json"):
        src = os.path.join(a.ckpts[0], f)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(a.out, f))
    print("soup written to", a.out)


if __name__ == "__main__":
    main()
