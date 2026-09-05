#!/usr/bin/env python3
"""Uniform weight average (model soup) of checkpoints that share a parent.

All inputs must be full fine-tunes descended from the same weights, so the
average stays in one basin. Writes a complete, loadable model directory with
the greedy generation_config adopted in exp-03.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil

import torch
from safetensors.torch import load_file, save_file

BASE = "/home/ben/hf_cache/hub/models--google--gemma-3-4b-pt/snapshots/cc012e0a6d0787b4adcc0fa2c4da74402494554d"
SIDE_FILES = (
    "config.json",
    "model.safetensors.index.json",
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "preprocessor_config.json",
    "processor_config.json",
    "chat_template.jinja",
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", nargs="+", required=True)
    ap.add_argument("--weights", nargs="*", type=float, default=None)
    ap.add_argument("--dst", required=True)
    args = ap.parse_args()

    w = args.weights or [1.0 / len(args.src)] * len(args.src)
    assert len(w) == len(args.src)
    tot = sum(w)
    w = [x / tot for x in w]
    print("souping", list(zip(args.src, w)))

    os.makedirs(args.dst, exist_ok=True)
    ref = args.src[0]
    index = json.load(open(os.path.join(ref, "model.safetensors.index.json")))
    shards = sorted(set(index["weight_map"].values()))

    for shard in shards:
        acc = None
        for path, wi in zip(args.src, w):
            sd = load_file(os.path.join(path, shard))
            if acc is None:
                acc = {k: v.to(torch.float32) * wi for k, v in sd.items()}
            else:
                assert set(sd) == set(acc), f"key mismatch in {shard}"
                for k in acc:
                    acc[k] += sd[k].to(torch.float32) * wi
            del sd
        out = {k: v.to(torch.bfloat16) for k, v in acc.items()}
        save_file(out, os.path.join(args.dst, shard), metadata={"format": "pt"})
        print("wrote", shard, len(out), "tensors")
        del acc, out

    for fn in SIDE_FILES:
        for cand in (ref, BASE):
            src = os.path.join(cand, fn)
            if os.path.exists(src):
                shutil.copy2(src, os.path.join(args.dst, fn))
                break
    gc = {
        "bos_token_id": 2,
        "eos_token_id": [1, 106],
        "pad_token_id": 0,
        "cache_implementation": "hybrid",
        "do_sample": False,
        "temperature": 0.0,
    }
    json.dump(gc, open(os.path.join(args.dst, "generation_config.json"), "w"), indent=2)
    print("done ->", args.dst, sorted(os.listdir(args.dst)))


if __name__ == "__main__":
    main()
