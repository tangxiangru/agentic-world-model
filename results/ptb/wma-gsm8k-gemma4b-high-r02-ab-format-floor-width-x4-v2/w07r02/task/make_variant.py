#!/usr/bin/env python3
"""Create a checkpoint directory that shares weights with a source checkpoint
but serves a different generation_config.json.

evaluate.py passes no decode parameters, so vLLM falls back to the served
model's generation_config.json (vllm/config/model.py get_diff_sampling_param).
This is how a decode-config change is made testable without touching evaluate.py.
"""
from __future__ import annotations

import argparse
import json
import os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--temperature", type=float, default=None)
    ap.add_argument("--top-p", type=float, default=None)
    ap.add_argument("--top-k", type=int, default=None)
    ap.add_argument("--copy", action="store_true", help="copy weights instead of symlinking")
    args = ap.parse_args()

    src = os.path.abspath(args.src)
    dst = os.path.abspath(args.dst)
    os.makedirs(dst, exist_ok=True)
    for f in sorted(os.listdir(src)):
        if f == "generation_config.json":
            continue
        s, d = os.path.join(src, f), os.path.join(dst, f)
        if os.path.exists(d) or os.path.islink(d):
            continue
        if os.path.isdir(s):
            continue
        if args.copy:
            import shutil
            shutil.copy(s, d)
        else:
            os.symlink(os.path.realpath(s), d)

    gc = json.load(open(os.path.join(src, "generation_config.json")))
    for key, val in (("temperature", args.temperature), ("top_p", args.top_p), ("top_k", args.top_k)):
        if val is None:
            gc.pop(key, None)
        else:
            gc[key] = val
    gc["do_sample"] = bool(args.temperature)
    json.dump(gc, open(os.path.join(dst, "generation_config.json"), "w"), indent=2)
    print("wrote", dst)
    print(json.dumps(gc, indent=2))


if __name__ == "__main__":
    main()
