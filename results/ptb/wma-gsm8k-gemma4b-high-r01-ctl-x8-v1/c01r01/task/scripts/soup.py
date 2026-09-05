#!/usr/bin/env python3
"""Uniform weight average of two or more checkpoints of the same architecture."""
import argparse
import json
import os
import shutil

import torch
from safetensors.torch import load_file, save_file

ap = argparse.ArgumentParser()
ap.add_argument("--inputs", nargs="+", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()

os.makedirs(a.out, exist_ok=True)
ref = a.inputs[0]
index = json.load(open(os.path.join(ref, "model.safetensors.index.json")))
shards = sorted(set(index["weight_map"].values()))
for f in shards:
    acc = None
    for d in a.inputs:
        sd = load_file(os.path.join(d, f))
        if acc is None:
            acc = {k: v.to(torch.float32) for k, v in sd.items()}
        else:
            for k in acc:
                acc[k] += sd[k].to(torch.float32)
        del sd
    n = len(a.inputs)
    out = {k: (v / n).to(torch.bfloat16) for k, v in acc.items()}
    save_file(out, os.path.join(a.out, f), metadata={"format": "pt"})
    print("wrote", f, flush=True)
    del acc, out

for f in ["config.json", "generation_config.json", "model.safetensors.index.json",
          "added_tokens.json", "preprocessor_config.json", "processor_config.json",
          "special_tokens_map.json", "tokenizer.json", "tokenizer.model",
          "tokenizer_config.json"]:
    src = os.path.join(ref, f)
    if os.path.exists(src):
        shutil.copy(src, os.path.join(a.out, f))
print("soup written to", a.out)
