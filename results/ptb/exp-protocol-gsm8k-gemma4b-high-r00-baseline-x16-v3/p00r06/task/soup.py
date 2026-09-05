#!/usr/bin/env python3
"""Uniform weight average of checkpoints that share a parent (model soup)."""
from __future__ import annotations

import argparse
import json
import os
import shutil

import torch
from safetensors.torch import load_file, save_file
from transformers import AutoConfig

GREEDY = {
    "bos_token_id": 2,
    "eos_token_id": [1, 106],
    "pad_token_id": 0,
    "cache_implementation": "hybrid",
    "do_sample": False,
    "temperature": 0.0,
    "transformers_version": "4.57.3",
}
SIDE = (
    "config.json",
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "preprocessor_config.json",
    "processor_config.json",
    "model.safetensors.index.json",
)


def shard_files(d: str) -> list[str]:
    idx = os.path.join(d, "model.safetensors.index.json")
    if os.path.exists(idx):
        names = sorted(set(json.load(open(idx))["weight_map"].values()))
    else:
        names = ["model.safetensors"]
    return names


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpts", nargs="+")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    base = args.ckpts[0]
    names = shard_files(base)
    for other in args.ckpts[1:]:
        assert shard_files(other) == names, f"shard layout differs: {other}"

    for fn in SIDE:
        src = os.path.join(base, fn)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(args.out, fn))

    k = len(args.ckpts)
    for name in names:
        acc = None
        for d in args.ckpts:
            sd = load_file(os.path.join(d, name))
            if acc is None:
                acc = {t: v.to(torch.float32) for t, v in sd.items()}
            else:
                assert sd.keys() == acc.keys(), f"key mismatch in {d}/{name}"
                for t, v in sd.items():
                    acc[t] += v.to(torch.float32)
            del sd
        out = {t: (v / k).to(torch.bfloat16) for t, v in acc.items()}
        save_file(out, os.path.join(args.out, name), metadata={"format": "pt"})
        print("wrote", name, len(out), "tensors")
        del acc, out

    with open(os.path.join(args.out, "generation_config.json"), "w") as f:
        json.dump(GREEDY, f, indent=2)
    AutoConfig.from_pretrained(args.out)  # sanity: config still parses
    print("souped", k, "checkpoints ->", args.out)


if __name__ == "__main__":
    main()
