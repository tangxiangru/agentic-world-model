#!/usr/bin/env python3
"""Uniform weight average (model soup) of checkpoints fine-tuned from the same base."""
import argparse
import os
import shutil

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--ckpts", nargs="+", required=True)
ap.add_argument("--out", required=True)
args = ap.parse_args()

acc = None
for i, c in enumerate(args.ckpts):
    m = AutoModelForCausalLM.from_pretrained(c, dtype=torch.float32, device_map="cpu")
    sd = m.state_dict()
    if acc is None:
        acc = {k: v.clone() for k, v in sd.items()}
    else:
        assert acc.keys() == sd.keys()
        for k in acc:
            acc[k] += sd[k]
    del m, sd
    print("added", c, flush=True)

n = len(args.ckpts)
for k in acc:
    acc[k] /= n

model = AutoModelForCausalLM.from_pretrained(args.ckpts[0], dtype=torch.float32,
                                             device_map="cpu")
model.load_state_dict(acc)
model = model.to(torch.bfloat16)
os.makedirs(args.out, exist_ok=True)
model.save_pretrained(args.out, safe_serialization=True)
AutoTokenizer.from_pretrained(args.ckpts[0]).save_pretrained(args.out)
for extra in ("preprocessor_config.json", "processor_config.json",
              "generation_config.json"):
    s = os.path.join(args.ckpts[0], extra)
    if os.path.exists(s):
        shutil.copy(s, os.path.join(args.out, extra))
print("saved soup ->", args.out)
