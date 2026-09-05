#!/usr/bin/env python3
"""Make a checkpoint variant that differs only in generation_config.json.

Weights are symlinked, so a variant costs no disk and no copy time. Used to
A/B the decode config the grader inherits from the model (vLLM overrides its
own defaults with the model's generation_config).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil

ap = argparse.ArgumentParser()
ap.add_argument("--src", required=True)
ap.add_argument("--dst", required=True)
ap.add_argument("--set", nargs="*", default=[], help="key=value json fragments")
ap.add_argument("--drop", nargs="*", default=[])
args = ap.parse_args()

os.makedirs(args.dst, exist_ok=True)
for fn in os.listdir(args.src):
    s, d = os.path.abspath(os.path.join(args.src, fn)), os.path.join(args.dst, fn)
    if os.path.lexists(d):
        os.remove(d)
    if fn == "generation_config.json":
        shutil.copy(s, d)
    else:
        os.symlink(s, d)

p = os.path.join(args.dst, "generation_config.json")
cfg = json.load(open(p))
for kv in args.set:
    k, _, v = kv.partition("=")
    cfg[k] = json.loads(v)
for k in args.drop:
    cfg.pop(k, None)
json.dump(cfg, open(p, "w"), indent=2)
print(json.dumps(cfg, indent=2))
print("wrote", args.dst)
