#!/usr/bin/env python3
"""Clone a checkpoint dir with a different generation_config.json.

inspect sends no temperature/top_p, so vLLM (generation_config="auto") takes its
sampling defaults from the model directory. Decoding is therefore a property of
the checkpoint, and swapping it is a decode-config experiment on hardlinked
weights -- no copy of the 8.6 GB of safetensors.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--greedy", action="store_true")
    ap.add_argument("--temperature", type=float, default=None)
    ap.add_argument("--top-p", type=float, default=None)
    ap.add_argument("--top-k", type=int, default=None)
    args = ap.parse_args()

    os.makedirs(args.dst, exist_ok=True)
    for f in os.listdir(args.src):
        s, d = os.path.join(args.src, f), os.path.join(args.dst, f)
        if os.path.exists(d):
            os.remove(d)
        if f == "generation_config.json" or os.path.isdir(s):
            continue
        try:
            os.link(s, d)
        except OSError:
            shutil.copy(s, d)

    gen = json.load(open(os.path.join(args.src, "generation_config.json")))
    if args.greedy:
        gen.update({"do_sample": False, "temperature": 0.0, "top_p": 1.0, "top_k": 0})
    if args.temperature is not None:
        gen["temperature"] = args.temperature
        gen["do_sample"] = args.temperature > 0
    if args.top_p is not None:
        gen["top_p"] = args.top_p
    if args.top_k is not None:
        gen["top_k"] = args.top_k
    json.dump(gen, open(os.path.join(args.dst, "generation_config.json"), "w"), indent=2)
    print(json.dumps(gen), "->", args.dst)


if __name__ == "__main__":
    main()
