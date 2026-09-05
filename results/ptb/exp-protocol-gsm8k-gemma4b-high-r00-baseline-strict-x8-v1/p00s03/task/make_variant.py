#!/usr/bin/env python3
"""Create a checkpoint directory that differs from its source only in
generation_config.json (the decoding defaults vLLM picks up for requests that
do not set temperature/top_p/top_k).  Weights are symlinked, so a variant
costs no disk.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil

ap = argparse.ArgumentParser()
ap.add_argument("--src", required=True)
ap.add_argument("--dst", required=True)
ap.add_argument("--temperature", type=float, default=None)
ap.add_argument("--top-p", type=float, default=None)
ap.add_argument("--top-k", type=int, default=None)
ap.add_argument("--copy", action="store_true", help="copy weights instead of symlinking")
ap.add_argument("--tok-src", default="/home/ben/task/ckpts/exp-04/final")
a = ap.parse_args()

os.makedirs(a.dst, exist_ok=True)
for fn in os.listdir(a.src):
    s, d = os.path.join(a.src, fn), os.path.join(a.dst, fn)
    if os.path.lexists(d):
        os.remove(d)
    if fn in ("generation_config.json", "optimizer.pt", "rng_state.pth",
              "scheduler.pt", "trainer_state.json"):
        continue
    if fn.endswith(".safetensors") and not a.copy:
        os.symlink(os.path.abspath(s), d)
    else:
        shutil.copy(s, d)

# intermediate Trainer checkpoints have no tokenizer; take it from --tok-src
for fn in os.listdir(a.tok_src):
    if fn.endswith(".safetensors") or fn.endswith(".json") and fn in ("config.json",):
        continue
    if "token" in fn or fn in ("special_tokens_map.json", "added_tokens.json",
                               "preprocessor_config.json", "processor_config.json",
                               "chat_template.jinja"):
        d = os.path.join(a.dst, fn)
        if not os.path.exists(d):
            shutil.copy(os.path.join(a.tok_src, fn), d)

gc = json.load(open(os.path.join(a.src, "generation_config.json")))
for k, v in (("temperature", a.temperature), ("top_p", a.top_p), ("top_k", a.top_k)):
    if v is not None:
        gc[k] = v
gc["do_sample"] = not (a.temperature == 0.0 or a.top_k == 1)
json.dump(gc, open(os.path.join(a.dst, "generation_config.json"), "w"), indent=2)
print(a.dst, json.dumps(gc))
