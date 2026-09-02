#!/usr/bin/env python3
"""Create a model directory that is the same weights with a different generation_config.

Weights are symlinked (vLLM/transformers follow symlinks), everything else is copied,
then generation_config.json is rewritten with the requested decoding defaults.

vLLM 0.11 runs with --generation-config auto, so the sampling fields in this file are
the server's defaults for any request that does not set them -- and evaluate.py sets
only max_tokens, so this file decides the decode.
"""
import argparse
import json
import os
import shutil


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--greedy", action="store_true", help="temperature 0, no top_k/top_p")
    ap.add_argument("--temperature", type=float, default=None)
    ap.add_argument("--copy", action="store_true", help="real copies instead of symlinks")
    args = ap.parse_args()

    os.makedirs(args.dst, exist_ok=True)
    for fn in os.listdir(args.src):
        src = os.path.join(args.src, fn)
        dst = os.path.join(args.dst, fn)
        if os.path.isdir(src):
            continue
        if os.path.lexists(dst):
            os.remove(dst)
        if fn.endswith(".safetensors") and not args.copy:
            os.symlink(os.path.realpath(src), dst)
        else:
            shutil.copy(src, dst)

    gc_path = os.path.join(args.dst, "generation_config.json")
    gc = json.load(open(gc_path))
    if args.greedy:
        gc["do_sample"] = False
        gc["temperature"] = 0.0
        gc.pop("top_k", None)
        gc.pop("top_p", None)
    if args.temperature is not None:
        gc["do_sample"] = args.temperature > 0
        gc["temperature"] = args.temperature
    json.dump(gc, open(gc_path, "w"), indent=2)
    print(json.dumps(gc, indent=1))
    print("wrote", args.dst)


if __name__ == "__main__":
    main()
