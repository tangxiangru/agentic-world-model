#!/usr/bin/env python3
"""Uniform weight average ("model soup") of two or more checkpoints of the same
architecture. Writes a checkpoint dir the grader can load unchanged.
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
    if os.path.exists(idx):
        names = sorted(set(json.load(open(idx))["weight_map"].values()))
        return [os.path.join(d, n) for n in names]
    return [os.path.join(d, "model.safetensors")]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", action="append", required=True)
    ap.add_argument("--weight", action="append", type=float, default=None,
                    help="one per --src; defaults to uniform. Normalised to sum 1.")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    ref = args.src[0]
    files = shard_files(ref)
    others = [shard_files(s) for s in args.src[1:]]
    for s_others in others:
        assert len(s_others) == len(files), "checkpoints have different shard layouts"

    w = args.weight or [1.0] * len(args.src)
    assert len(w) == len(args.src), "--weight must be given once per --src"
    tot = sum(w)
    w = [x / tot for x in w]
    print("weights", dict(zip(args.src, w)))
    for i, f in enumerate(files):
        acc = load_file(f)
        for k in acc:
            acc[k] = acc[k].to(torch.float32) * w[0]
        for j, s_others in enumerate(others):
            other = load_file(s_others[i])
            assert set(other) == set(acc), "key mismatch between checkpoints"
            for k in acc:
                acc[k] += other[k].to(torch.float32) * w[j + 1]
            del other
        for k in acc:
            acc[k] = acc[k].to(torch.bfloat16)
        out_f = os.path.join(args.out, os.path.basename(f))
        save_file(acc, out_f, metadata={"format": "pt"})
        print("wrote", out_f, flush=True)
        del acc

    for extra in ("model.safetensors.index.json", "config.json",
                  "generation_config.json", "tokenizer.json", "tokenizer.model",
                  "tokenizer_config.json", "special_tokens_map.json",
                  "added_tokens.json", "preprocessor_config.json",
                  "processor_config.json"):
        s = os.path.join(ref, extra)
        if os.path.exists(s):
            shutil.copy2(s, os.path.join(args.out, extra))
    print("souped", args.src, "->", args.out)


if __name__ == "__main__":
    main()
