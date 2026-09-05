#!/usr/bin/env python3
"""Uniform weight average of several checkpoints (same architecture/trajectory)."""
import argparse, os, shutil, json, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE = os.path.expanduser(
    "~/hf_cache/hub/models--google--gemma-3-4b-pt/snapshots/cc012e0a6d0787b4adcc0fa2c4da74402494554d")

ap = argparse.ArgumentParser()
ap.add_argument("--srcs", nargs="+", required=True)
ap.add_argument("--dst", required=True)
a = ap.parse_args()

acc = None
for i, s in enumerate(a.srcs):
    m = AutoModelForCausalLM.from_pretrained(s, dtype=torch.float32)
    sd = m.state_dict()
    if acc is None:
        acc = {k: v.clone() for k, v in sd.items()}
        base_model = m
    else:
        for k in acc:
            acc[k] += sd[k]
        del m
    print("added", s)
n = len(a.srcs)
for k in acc:
    acc[k] /= n
base_model.load_state_dict(acc)
base_model = base_model.to(torch.bfloat16)
base_model.config.torch_dtype = "bfloat16"
if hasattr(base_model.config, "text_config"):
    base_model.config.text_config.torch_dtype = "bfloat16"
base_model.config.use_cache = True
from transformers import GenerationConfig
base_model.generation_config = GenerationConfig(
    bos_token_id=2, eos_token_id=[1, 106], pad_token_id=0, cache_implementation="hybrid")
os.makedirs(a.dst, exist_ok=True)
base_model.save_pretrained(a.dst, safe_serialization=True)
AutoTokenizer.from_pretrained(BASE).save_pretrained(a.dst)
for f in ["preprocessor_config.json", "processor_config.json"]:
    p = os.path.join(BASE, f)
    if os.path.exists(p):
        shutil.copy(p, os.path.join(a.dst, f))
with open(os.path.join(a.dst, "generation_config.json"), "w") as f:
    json.dump({"bos_token_id": 2, "eos_token_id": [1, 106], "pad_token_id": 0,
               "cache_implementation": "hybrid", "do_sample": False,
               "temperature": 0.0, "top_p": 1.0, "top_k": -1}, f, indent=2)
print("soup ->", a.dst)
