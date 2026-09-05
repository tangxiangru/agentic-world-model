#!/usr/bin/env python3
"""Clone a checkpoint dir as symlinks, overriding only generation_config.json.

vLLM applies the model's generation_config.json as its default sampling params
(ModelConfig.get_diff_sampling_param), so decoding is a property of the saved
model, not of evaluate.py. This makes that one file the intervention.
"""
from __future__ import annotations

import argparse
import json
import os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--top-k", type=int, default=0)
    args = ap.parse_args()

    src = os.path.abspath(args.src)
    dst = os.path.abspath(args.dst)
    os.makedirs(dst, exist_ok=True)
    for fn in os.listdir(src):
        if fn == "generation_config.json":
            continue
        d = os.path.join(dst, fn)
        if os.path.lexists(d):
            os.remove(d)
        os.symlink(os.path.join(src, fn), d)

    gc = {
        "bos_token_id": 2,
        "eos_token_id": [1, 106],
        "pad_token_id": 0,
        "cache_implementation": "hybrid",
        "do_sample": args.temperature > 0,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
    }
    with open(os.path.join(dst, "generation_config.json"), "w") as f:
        json.dump(gc, f, indent=2)
    print(json.dumps({"dst": dst, "generation_config": gc}, indent=2))


if __name__ == "__main__":
    main()
