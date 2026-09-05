#!/usr/bin/env python3
"""Create a checkpoint directory that shares another checkpoint's weights but
carries a different generation_config.json (decode-config experiments).

Weights are symlinked, so a variant costs kilobytes. vLLM resolves symlinks.
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
    ap.add_argument("--set", nargs="*", default=[], help="key=value pairs for generation_config")
    ap.add_argument("--drop", nargs="*", default=[], help="keys to remove from generation_config")
    ap.add_argument("--copy", action="store_true", help="copy weights instead of symlinking")
    args = ap.parse_args()

    os.makedirs(args.dst, exist_ok=True)
    for name in os.listdir(args.src):
        s, d = os.path.join(args.src, name), os.path.join(args.dst, name)
        if os.path.lexists(d):
            os.remove(d)
        if name == "generation_config.json":
            continue
        if args.copy or not name.endswith(".safetensors"):
            shutil.copy2(os.path.realpath(s), d)
        else:
            os.symlink(os.path.realpath(s), d)

    gc = json.load(open(os.path.join(args.src, "generation_config.json")))
    for k in args.drop:
        gc.pop(k, None)
    for kv in args.set:
        k, v = kv.split("=", 1)
        try:
            v = json.loads(v)
        except json.JSONDecodeError:
            pass
        gc[k] = v
    with open(os.path.join(args.dst, "generation_config.json"), "w") as f:
        json.dump(gc, f, indent=2)
    print(json.dumps(gc, indent=1))
    print("wrote", args.dst)


if __name__ == "__main__":
    main()
