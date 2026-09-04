#!/usr/bin/env python3
"""Create a scratch model dir that shares the parent's weights (symlinks) but ships a
different generation_config.json. vLLM reads temperature/top_p/top_k from that file, so
this is how a decode change is made without touching the trained weights."""
import argparse
import json
import os

ap = argparse.ArgumentParser()
ap.add_argument("--src", required=True)
ap.add_argument("--dst", required=True)
ap.add_argument("--temperature", type=float, default=0.0)
ap.add_argument("--keep-sampling", action="store_true")
a = ap.parse_args()

os.makedirs(a.dst, exist_ok=True)
for f in os.listdir(a.src):
    if f == "generation_config.json":
        continue
    s, d = os.path.abspath(os.path.join(a.src, f)), os.path.join(a.dst, f)
    if os.path.isdir(s):
        continue
    if os.path.lexists(d):
        os.remove(d)
    os.symlink(s, d)

gc = json.load(open(os.path.join(a.src, "generation_config.json")))
if not a.keep_sampling:
    gc.pop("top_k", None)
    gc.pop("top_p", None)
    gc["do_sample"] = False
gc["temperature"] = a.temperature
json.dump(gc, open(os.path.join(a.dst, "generation_config.json"), "w"), indent=2)
print(json.dumps(gc, indent=2))
print("wrote", a.dst)
