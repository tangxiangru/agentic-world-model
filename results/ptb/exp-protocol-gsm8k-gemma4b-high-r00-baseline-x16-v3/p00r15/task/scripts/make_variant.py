#!/usr/bin/env python3
"""Make a decode-config variant of a checkpoint: same weights (symlinked), new
generation_config.json.

vLLM 0.11 defaults to `--generation-config auto`, so the sampling params in a
checkpoint's generation_config.json become the server's defaults. The stock
gemma-3-4b-pt config asks for do_sample/top_k=64/top_p=0.95, i.e. temperature-1
sampling; that is a decode choice this task lets us make, not part of evaluate.py.
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
    ap.add_argument("--copy", action="store_true", help="copy weights instead of symlinking")
    args = ap.parse_args()

    os.makedirs(args.dst, exist_ok=True)
    for fn in sorted(os.listdir(args.src)):
        src = os.path.join(args.src, fn)
        dst = os.path.join(args.dst, fn)
        if os.path.lexists(dst):
            os.remove(dst)
        if fn == "generation_config.json":
            continue
        if fn.endswith(".safetensors") and not args.copy:
            os.symlink(os.path.realpath(src), dst)
        elif os.path.isfile(src):
            shutil.copy(src, dst)

    gc = json.load(open(os.path.join(args.src, "generation_config.json")))
    if args.greedy:
        gc["do_sample"] = False
        gc["temperature"] = 0.0
        gc.pop("top_p", None)
        gc.pop("top_k", None)
    if args.temperature is not None:
        gc["temperature"] = args.temperature
        gc["do_sample"] = args.temperature > 0
    if args.top_p is not None:
        gc["top_p"] = args.top_p
    if args.top_k is not None:
        gc["top_k"] = args.top_k
    json.dump(gc, open(os.path.join(args.dst, "generation_config.json"), "w"), indent=2)
    print("wrote", args.dst, json.dumps(gc))


if __name__ == "__main__":
    main()
