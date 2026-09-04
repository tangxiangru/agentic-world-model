#!/usr/bin/env python3
"""Uniform weight average (model soup) of several checkpoints of the same architecture."""
import argparse, json, os, shutil

import torch
from safetensors.torch import load_file, save_file
from transformers import AutoTokenizer, Gemma3ForConditionalGeneration

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", action="append", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()

acc = None
for i, c in enumerate(a.ckpt):
    idx = json.load(open(os.path.join(c, "model.safetensors.index.json")))
    files = sorted(set(idx["weight_map"].values()))
    sd = {}
    for f in files:
        sd.update(load_file(os.path.join(c, f)))
    print(f"{c}: {len(sd)} tensors")
    if acc is None:
        acc = {k: v.to(torch.float32) for k, v in sd.items()}
    else:
        assert set(acc) == set(sd), "checkpoints disagree on parameter names"
        for k in acc:
            acc[k] += sd[k].to(torch.float32)
    del sd

n = len(a.ckpt)
for k in acc:
    acc[k] = (acc[k] / n).to(torch.bfloat16)

# Write the shards by hand with the source checkpoint's exact file layout and key names.
# save_pretrained() re-applies gemma-3's save-time key rename to an already-renamed state
# dict, which produces 'language_model.language_model.*' and a model vLLM cannot load.
os.makedirs(a.out, exist_ok=True)
src = a.ckpt[0]
idx = json.load(open(os.path.join(src, "model.safetensors.index.json")))
by_file = {}
for k, f in idx["weight_map"].items():
    by_file.setdefault(f, {})[k] = acc[k]
for f, tensors in by_file.items():
    save_file(tensors, os.path.join(a.out, f), metadata={"format": "pt"})
    print("wrote", f, len(tensors))
for f in os.listdir(src):
    if f.endswith(".safetensors"):
        continue
    shutil.copyfile(os.path.join(src, f), os.path.join(a.out, f))
# the source configs declare float32 (save_pretrained ignored the torch_dtype we set); the
# tensors are bf16, so say so, and carry the greedy decode config adopted in exp-03
cfg = json.load(open(os.path.join(a.out, "config.json")))
cfg["dtype"] = "bfloat16"
cfg.get("text_config", {})["dtype"] = "bfloat16"
cfg.get("vision_config", {})["dtype"] = "bfloat16"
json.dump(cfg, open(os.path.join(a.out, "config.json"), "w"), indent=2)
json.dump({"bos_token_id": 2, "cache_implementation": "hybrid", "eos_token_id": [1, 106],
           "pad_token_id": 0, "temperature": 0.0, "top_k": -1, "top_p": 1.0},
          open(os.path.join(a.out, "generation_config.json"), "w"), indent=2)
print("wrote", a.out)
