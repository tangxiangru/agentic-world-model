#!/usr/bin/env python3
"""Merge a GRPO LoRA adapter into the SFT weights and write a standalone model."""
from __future__ import annotations

import argparse
import os
import shutil

import torch
from peft import PeftModel
from transformers import AutoTokenizer, Gemma3ForConditionalGeneration

from train_sft import BASE

ap = argparse.ArgumentParser()
ap.add_argument("--base", required=True, help="SFT model the adapter was trained on")
ap.add_argument("--adapter", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()

model = Gemma3ForConditionalGeneration.from_pretrained(a.base, dtype=torch.bfloat16)
model = PeftModel.from_pretrained(model, a.adapter, dtype=torch.bfloat16)
model = model.merge_and_unload()
model.config.use_cache = True
os.makedirs(a.out, exist_ok=True)
model.save_pretrained(a.out, safe_serialization=True)
AutoTokenizer.from_pretrained(BASE).save_pretrained(a.out)
for fn in ("preprocessor_config.json", "processor_config.json"):
    src = os.path.join(BASE, fn)
    if os.path.exists(src):
        shutil.copy(src, os.path.join(a.out, fn))
print("merged ->", a.out)
