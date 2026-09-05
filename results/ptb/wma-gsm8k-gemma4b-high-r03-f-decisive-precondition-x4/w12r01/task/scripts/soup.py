#!/usr/bin/env python3
"""Uniform weight average of two or more checkpoints of the same architecture."""
import argparse
import json
import os
import shutil

import torch
from safetensors.torch import load_file, save_file

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", action="append", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--template", required=True, help="dir to copy config/tokenizer/processor from")
a = ap.parse_args()

os.makedirs(a.out, exist_ok=True)
idx = json.load(open(os.path.join(a.ckpt[0], "model.safetensors.index.json")))
shards = sorted(set(idx["weight_map"].values()))
n = len(a.ckpt)
for shard in shards:
    acc = None
    for c in a.ckpt:
        sd = load_file(os.path.join(c, shard))
        if acc is None:
            acc = {k: v.to(torch.float32) for k, v in sd.items()}
        else:
            for k in acc:
                acc[k] += sd[k].to(torch.float32)
        del sd
    out = {k: (v / n).to(torch.bfloat16) for k, v in acc.items()}
    save_file(out, os.path.join(a.out, shard), metadata={"format": "pt"})
    print("wrote", shard, flush=True)
    del acc, out

for f in ("model.safetensors.index.json", "config.json", "generation_config.json",
          "tokenizer.json", "tokenizer.model", "tokenizer_config.json",
          "special_tokens_map.json", "added_tokens.json",
          "preprocessor_config.json", "processor_config.json"):
    src = os.path.join(a.template, f)
    if os.path.exists(src):
        shutil.copyfile(src, os.path.join(a.out, f))
print("souped", n, "checkpoints ->", a.out)
