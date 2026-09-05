#!/usr/bin/env python3
"""Cast a saved fp32 checkpoint to bf16 for serving (the base snapshot is bf16).

Keeps architecture, tokenizer and processor files identical.
"""
import argparse
import os
import shutil

import torch
from transformers import AutoTokenizer, Gemma3ForConditionalGeneration

SNAPSHOT = "/home/ben/hf_cache/hub/models--google--gemma-3-4b-pt/snapshots/cc012e0a6d0787b4adcc0fa2c4da74402494554d"

ap = argparse.ArgumentParser()
ap.add_argument("--src", required=True)
ap.add_argument("--dst", required=True)
a = ap.parse_args()

m = Gemma3ForConditionalGeneration.from_pretrained(a.src, dtype=torch.bfloat16)
m.config.torch_dtype = "bfloat16"
m.config.use_cache = True
m.save_pretrained(a.dst)
tok = AutoTokenizer.from_pretrained(a.src if os.path.exists(os.path.join(a.src, "tokenizer.json")) else SNAPSHOT)
tok.save_pretrained(a.dst)
for f in ("preprocessor_config.json", "processor_config.json", "added_tokens.json"):
    src = os.path.join(a.src, f)
    if not os.path.exists(src):
        src = os.path.join(SNAPSHOT, f)
    if os.path.exists(src):
        shutil.copy(src, os.path.join(a.dst, f))
print("wrote", a.dst)
