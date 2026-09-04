#!/usr/bin/env python3
"""Uniform weight-space average of two checkpoints from the same lineage.

exp-05 is a continuation of exp-02, so the two sit in the same loss basin and a
parameter average is well defined. Everything non-tensor (config, tokenizer,
processor) is taken from the first parent.
"""
from __future__ import annotations

import argparse, json, os, shutil
import torch
from transformers import Gemma3ForConditionalGeneration, AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--a", required=True)
ap.add_argument("--b", required=True)
ap.add_argument("--weight-a", type=float, default=0.5)
ap.add_argument("--out", required=True)
args = ap.parse_args()

wa, wb = args.weight_a, 1.0 - args.weight_a
ma = Gemma3ForConditionalGeneration.from_pretrained(args.a, dtype=torch.bfloat16)
sb = Gemma3ForConditionalGeneration.from_pretrained(args.b, dtype=torch.bfloat16).state_dict()
sa = ma.state_dict()
assert set(sa) == set(sb), "state dicts differ in shape"
# embed_tokens.weight and lm_head.weight are two state_dict entries over ONE
# storage (tie_word_embeddings=True): distinct Python objects, equal data_ptr.
# Updating in place per key would apply the average twice and land that tensor
# on 0.25*a + 0.75*b instead of the uniform average, so dedupe by storage.
seen: set[int] = set()
n_diff = n_tied = 0
for k in sa:
    t = sa[k]
    if t.data_ptr() in seen:
        n_tied += 1
        continue
    seen.add(t.data_ptr())
    if not torch.equal(t, sb[k]):
        n_diff += 1
    t.mul_(wa).add_(sb[k].to(t.dtype), alpha=wb)
ma.load_state_dict(sa)
ma.save_pretrained(args.out, safe_serialization=True)
AutoTokenizer.from_pretrained(args.a).save_pretrained(args.out)
for f in ("preprocessor_config.json", "processor_config.json", "generation_config.json"):
    s = os.path.join(args.a, f)
    if os.path.exists(s):
        shutil.copy(s, os.path.join(args.out, f))
print(f"souped {wa}*{args.a} + {wb}*{args.b} -> {args.out}; "
      f"{len(seen)} distinct tensors averaged ({n_tied} tied aliases skipped), {n_diff} differed between parents")
