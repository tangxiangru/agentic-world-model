#!/usr/bin/env python3
"""Uniform weight-space average of two fine-tunes of the same base snapshot."""
import argparse
import subprocess
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--models", nargs="+", required=True)
ap.add_argument("--out", required=True)
args = ap.parse_args()

base = AutoModelForCausalLM.from_pretrained(args.models[0], dtype=torch.float32)
sd = base.state_dict()
# tied tensors (embed_tokens / lm_head share storage) must be accumulated once
seen, keys = set(), []
for k, v in sd.items():
    if v.data_ptr() in seen:
        print("tied, skipped:", k)
        continue
    seen.add(v.data_ptr())
    keys.append(k)
for extra in args.models[1:]:
    m = AutoModelForCausalLM.from_pretrained(extra, dtype=torch.float32)
    msd = m.state_dict()
    assert set(msd) == set(sd), "state dicts differ"
    for k in keys:
        sd[k] += msd[k]
    del m, msd
n = len(args.models)
for k in keys:
    sd[k] /= n
base.load_state_dict(sd)
base = base.to(torch.bfloat16)
base.config.use_cache = True
base.config.torch_dtype = torch.bfloat16
for sub in ("text_config", "vision_config"):
    if hasattr(base.config, sub):
        getattr(base.config, sub).torch_dtype = torch.bfloat16
# strip the greedy values before saving: transformers refuses do_sample=False
# together with temperature/top_k. fix_gen_config.py rewrites the file after.
for f in ("temperature", "top_p", "top_k"):
    setattr(base.generation_config, f, None)
base.generation_config.do_sample = False
base.save_pretrained(args.out, safe_serialization=True)
AutoTokenizer.from_pretrained(args.models[0]).save_pretrained(args.out)
subprocess.run([sys.executable, "/home/ben/task/fix_gen_config.py", args.out], check=True)
print("wrote", args.out)
