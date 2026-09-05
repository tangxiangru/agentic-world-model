#!/usr/bin/env python3
"""Make a decode-config variant of a checkpoint without copying the weights.

evaluate.py passes nothing about sampling to vLLM, and vLLM's default
--generation-config=auto means the model directory's generation_config.json IS
the decode configuration the grader uses ("Default sampling parameters have
been overridden by the model's Hugging Face generation config", exp-01 log).
The base snapshot ships do_sample=true / top_k=64 / top_p=0.95, so the harness
grades with temperature-1.0 sampling unless the checkpoint says otherwise.

This links every file of the source checkpoint into a new directory and writes
a different generation_config.json there, so two decode configs can be scored
under the same protocol without duplicating 8.6 GB.
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
    ap.add_argument("--mode", choices=["greedy", "sample"], default="greedy")
    ap.add_argument("--temperature", type=float, default=None)
    ap.add_argument("--top-p", type=float, default=None)
    ap.add_argument("--copy", action="store_true", help="real copies, not links")
    args = ap.parse_args()

    # symlink targets must be absolute: a relative target resolves against the
    # LINK's directory, not the cwd, and silently produces a dangling link
    args.src = os.path.abspath(args.src)
    os.makedirs(args.dst, exist_ok=True)
    for name in os.listdir(args.src):
        s, d = os.path.join(args.src, name), os.path.join(args.dst, name)
        if name == "generation_config.json" or os.path.exists(d):
            continue
        if os.path.isdir(s):
            continue
        (shutil.copy2 if args.copy else os.symlink)(s, d)

    gc = json.load(open(os.path.join(args.src, "generation_config.json")))
    if args.mode == "greedy":
        gc.update({"do_sample": False, "temperature": 0.0, "top_p": 1.0, "top_k": 0})
    else:
        gc["do_sample"] = True
        if args.temperature is not None:
            gc["temperature"] = args.temperature
        if args.top_p is not None:
            gc["top_p"] = args.top_p
    # <end_of_turn> must stay a stop token whatever the decode config
    eos = gc.get("eos_token_id")
    eos = [eos] if isinstance(eos, int) else list(eos or [])
    for t in (1, 106):
        if t not in eos:
            eos.append(t)
    gc["eos_token_id"] = eos
    json.dump(gc, open(os.path.join(args.dst, "generation_config.json"), "w"), indent=2)
    print(json.dumps(gc, indent=2))
    print("wrote", args.dst)


if __name__ == "__main__":
    main()
