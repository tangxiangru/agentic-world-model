#!/usr/bin/env python3
"""Merge the LoRA adapter into the base weights and write a standalone final_model/.

Guards the `final_model_not_loadable` pitfall: merged weights (not an adapter dir),
tokenizer saved alongside, same Gemma3ForConditionalGeneration architecture vLLM expects.
"""
import argparse
import shutil
import os
import torch
from transformers import AutoTokenizer, AutoProcessor, Gemma3ForConditionalGeneration
from peft import PeftModel

SNAP = "/home/ben/hf_cache/hub/models--google--gemma-3-4b-pt/snapshots/cc012e0a6d0787b4adcc0fa2c4da74402494554d"

ap = argparse.ArgumentParser()
ap.add_argument("--adapter", required=True)
ap.add_argument("--out", default="final_model")
a = ap.parse_args()

print("loading base...")
model = Gemma3ForConditionalGeneration.from_pretrained(SNAP, dtype=torch.bfloat16)
print("applying adapter", a.adapter)
model = PeftModel.from_pretrained(model, a.adapter)
model = model.merge_and_unload()
model.config.use_cache = True

os.makedirs(a.out, exist_ok=True)
model.save_pretrained(a.out, safe_serialization=True)

tok = AutoTokenizer.from_pretrained(SNAP)
tok.save_pretrained(a.out)
try:
    AutoProcessor.from_pretrained(SNAP).save_pretrained(a.out)
except Exception as e:
    print("processor copy skipped:", e)
    for fn in ("preprocessor_config.json", "processor_config.json"):
        src = os.path.join(SNAP, fn)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(a.out, fn))

print("wrote", a.out, sorted(os.listdir(a.out)))
