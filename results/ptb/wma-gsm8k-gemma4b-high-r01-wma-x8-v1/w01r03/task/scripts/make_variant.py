#!/usr/bin/env python3
"""Create a decode-config variant of a checkpoint.

The weights are symlinked (no copy), only generation_config.json is rewritten.
vLLM reads generation_config.json from the model directory and uses it as the
default sampling params for requests that do not set them - which is exactly
what inspect_ai does (its GenerateConfig leaves temperature/top_p unset).
"""
from __future__ import annotations

import argparse
import json
import os


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--temperature", type=float, default=None)
    ap.add_argument("--top-p", type=float, default=None)
    ap.add_argument("--top-k", type=int, default=None)
    ap.add_argument("--drop-sampling", action="store_true",
                    help="remove top_k/top_p/do_sample entirely")
    args = ap.parse_args()

    os.makedirs(args.dst, exist_ok=True)
    for fn in os.listdir(args.src):
        if fn == "generation_config.json":
            continue
        dst = os.path.join(args.dst, fn)
        if os.path.lexists(dst):
            os.remove(dst)
        os.symlink(os.path.realpath(os.path.join(args.src, fn)), dst)

    gc = json.load(open(os.path.join(args.src, "generation_config.json")))
    if args.drop_sampling:
        for k in ("top_k", "top_p", "do_sample"):
            gc.pop(k, None)
    if args.temperature is not None:
        gc["temperature"] = args.temperature
        gc["do_sample"] = args.temperature > 0
    if args.top_p is not None:
        gc["top_p"] = args.top_p
    if args.top_k is not None:
        gc["top_k"] = args.top_k
    json.dump(gc, open(os.path.join(args.dst, "generation_config.json"), "w"), indent=2)
    print(json.dumps(gc, indent=2))
    print("wrote", args.dst)


if __name__ == "__main__":
    main()
