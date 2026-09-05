#!/usr/bin/env python3
"""Re-save a checkpoint in bfloat16 (vLLM otherwise loads 17 GB of fp32 weights).
Optionally rewrite generation_config.json sampling defaults."""
import argparse, json, os, shutil, torch
from transformers import AutoTokenizer, Gemma3ForConditionalGeneration

ap = argparse.ArgumentParser()
ap.add_argument("--src", required=True)
ap.add_argument("--dst", required=True)
ap.add_argument("--temperature", type=float, default=None,
                help="if set, write this temperature into generation_config.json")
ap.add_argument("--greedy", action="store_true",
                help="temperature 0.0, top_k 1, top_p 1.0, do_sample false")
a = ap.parse_args()

m = Gemma3ForConditionalGeneration.from_pretrained(a.src, dtype=torch.bfloat16)
os.makedirs(a.dst, exist_ok=True)
m.save_pretrained(a.dst, safe_serialization=True)
AutoTokenizer.from_pretrained(a.src).save_pretrained(a.dst)
for f in ("preprocessor_config.json", "processor_config.json", "added_tokens.json"):
    p = os.path.join(a.src, f)
    if os.path.exists(p):
        shutil.copy(p, os.path.join(a.dst, f))

gc_path = os.path.join(a.dst, "generation_config.json")
gc = json.load(open(gc_path)) if os.path.exists(gc_path) else {}
if a.greedy:
    gc.update({"do_sample": False, "temperature": 0.0, "top_k": 1, "top_p": 1.0})
elif a.temperature is not None:
    gc["temperature"] = a.temperature
gc.setdefault("bos_token_id", 2)
gc.setdefault("eos_token_id", [1, 106])
gc.setdefault("pad_token_id", 0)
json.dump(gc, open(gc_path, "w"), indent=2)
print("wrote", a.dst, "generation_config:", gc)
