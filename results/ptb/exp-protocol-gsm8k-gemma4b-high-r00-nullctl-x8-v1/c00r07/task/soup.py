#!/usr/bin/env python3
"""Uniform weight average (model soup) of two checkpoints fine-tuned from the same base."""
import os
import shutil
import sys

import torch
from safetensors.torch import load_file, save_file

os.environ.setdefault("HF_HOME", "/home/ben/hf_cache")


def main():
    a, b, out = sys.argv[1], sys.argv[2], sys.argv[3]
    os.makedirs(out, exist_ok=True)
    import json
    idx = json.load(open(os.path.join(a, "model.safetensors.index.json")))
    shards = sorted(set(idx["weight_map"].values()))
    for sh in shards:
        wa = load_file(os.path.join(a, sh))
        wb = load_file(os.path.join(b, sh))
        assert set(wa) == set(wb), sh
        merged = {k: ((wa[k].to(torch.float32) + wb[k].to(torch.float32)) / 2).to(wa[k].dtype)
                  for k in wa}
        save_file(merged, os.path.join(out, sh), metadata={"format": "pt"})
        print("merged", sh, flush=True)
    for fn in os.listdir(a):
        if fn.endswith(".safetensors"):
            continue
        shutil.copy(os.path.join(a, fn), os.path.join(out, fn))
    print("soup written to", out)


if __name__ == "__main__":
    main()
