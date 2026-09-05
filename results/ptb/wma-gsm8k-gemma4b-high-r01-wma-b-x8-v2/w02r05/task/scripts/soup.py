#!/usr/bin/env python3
"""Uniform (or weighted) parameter average of two or more checkpoints, saved as
a loadable model dir with the greedy generation config."""
from __future__ import annotations

import argparse
import json
import os
import shutil

import torch
from safetensors.torch import load_file, save_file

BASE = "/home/ben/hf_cache/hub/models--google--gemma-3-4b-pt/snapshots/cc012e0a6d0787b4adcc0fa2c4da74402494554d"
SIDE = ["config.json", "tokenizer.json", "tokenizer.model", "tokenizer_config.json",
        "special_tokens_map.json", "added_tokens.json",
        "preprocessor_config.json", "processor_config.json"]


def shards(d):
    idx = json.load(open(os.path.join(d, "model.safetensors.index.json")))
    return idx, sorted(set(idx["weight_map"].values()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", nargs="+", required=True)
    ap.add_argument("--weights", nargs="*", type=float, default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    w = a.weights or [1.0 / len(a.src)] * len(a.src)
    assert len(w) == len(a.src)
    s = sum(w)
    w = [x / s for x in w]
    print("weights:", dict(zip(a.src, w)))

    os.makedirs(a.out, exist_ok=True)
    idx, files = shards(a.src[0])
    for fn in files:
        acc = None
        for src, wi in zip(a.src, w):
            t = load_file(os.path.join(src, fn))
            if acc is None:
                acc = {k: v.to(torch.float32) * wi for k, v in t.items()}
            else:
                for k in acc:
                    acc[k] += t[k].to(torch.float32) * wi
            del t
        save_file({k: v.to(torch.bfloat16) for k, v in acc.items()},
                  os.path.join(a.out, fn), metadata={"format": "pt"})
        print("wrote", fn, flush=True)
        del acc

    shutil.copy(os.path.join(a.src[0], "model.safetensors.index.json"),
                os.path.join(a.out, "model.safetensors.index.json"))
    for fn in SIDE:
        for d in (a.src[0], BASE):
            p = os.path.join(d, fn)
            if os.path.exists(p):
                shutil.copy(p, os.path.join(a.out, fn))
                break
    json.dump({"bos_token_id": 2, "eos_token_id": [1, 106], "pad_token_id": 0,
               "cache_implementation": "hybrid", "do_sample": False,
               "temperature": 0.0, "top_p": 1.0, "top_k": 0},
              open(os.path.join(a.out, "generation_config.json"), "w"), indent=2)
    print("saved", a.out)


if __name__ == "__main__":
    main()
