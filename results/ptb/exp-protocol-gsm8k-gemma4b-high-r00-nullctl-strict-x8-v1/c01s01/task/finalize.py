#!/usr/bin/env python3
"""Cast a saved checkpoint to bf16, copy processor files, force greedy generation config."""
import argparse, json, os, shutil, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE = os.path.expanduser(
    "~/hf_cache/hub/models--google--gemma-3-4b-pt/snapshots/cc012e0a6d0787b4adcc0fa2c4da74402494554d")

ap = argparse.ArgumentParser()
ap.add_argument("--src", required=True)
ap.add_argument("--dst", required=True)
ap.add_argument("--keep-sampling", action="store_true")
a = ap.parse_args()

m = AutoModelForCausalLM.from_pretrained(a.src, dtype=torch.bfloat16)
m.config.torch_dtype = "bfloat16"
if hasattr(m.config, "text_config"):
    m.config.text_config.torch_dtype = "bfloat16"
m.config.use_cache = True
os.makedirs(a.dst, exist_ok=True)
m.save_pretrained(a.dst, safe_serialization=True)
AutoTokenizer.from_pretrained(BASE).save_pretrained(a.dst)
for f in ["preprocessor_config.json", "processor_config.json"]:
    p = os.path.join(BASE, f)
    if os.path.exists(p):
        shutil.copy(p, os.path.join(a.dst, f))

gc = {"bos_token_id": 2, "eos_token_id": [1, 106], "pad_token_id": 0,
      "cache_implementation": "hybrid"}
if a.keep_sampling:
    gc.update({"do_sample": True, "top_k": 64, "top_p": 0.95})
else:
    gc.update({"do_sample": False, "temperature": 0.0, "top_p": 1.0, "top_k": -1})
with open(os.path.join(a.dst, "generation_config.json"), "w") as f:
    json.dump(gc, f, indent=2)
print("finalized ->", a.dst)
