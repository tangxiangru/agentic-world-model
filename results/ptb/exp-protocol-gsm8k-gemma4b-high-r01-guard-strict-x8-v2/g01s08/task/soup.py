#!/usr/bin/env python3
"""Uniform weight average of two checkpoints saved from the same parent."""
import argparse
import os
import shutil

import torch
from transformers import AutoTokenizer, Gemma3ForConditionalGeneration

ap = argparse.ArgumentParser()
ap.add_argument("--a", required=True)
ap.add_argument("--b", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--weight-a", type=float, default=0.5)
args = ap.parse_args()

ma = Gemma3ForConditionalGeneration.from_pretrained(args.a, dtype=torch.float32)
mb = Gemma3ForConditionalGeneration.from_pretrained(args.b, dtype=torch.float32)
sa, sb = ma.state_dict(), mb.state_dict()
assert set(sa) == set(sb), "state dicts differ"
w = args.weight_a
for k in sa:
    sa[k].mul_(w).add_(sb[k], alpha=1.0 - w)
ma.load_state_dict(sa)
ma = ma.to(torch.bfloat16)
ma.config.torch_dtype = "bfloat16"
ma.save_pretrained(args.out)
AutoTokenizer.from_pretrained(args.a).save_pretrained(args.out)
for f in ("preprocessor_config.json", "processor_config.json", "added_tokens.json",
          "generation_config.json"):
    src = os.path.join(args.a, f)
    if os.path.exists(src):
        shutil.copy(src, os.path.join(args.out, f))
print("wrote", args.out)
