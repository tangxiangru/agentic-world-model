#!/usr/bin/env python3
"""Uniform (or weighted) parameter average of checkpoints on one fine-tuning
trajectory. Saves bf16 with a greedy generation_config.json."""
import argparse, json, os, shutil, torch
from transformers import AutoTokenizer, Gemma3ForConditionalGeneration

ap = argparse.ArgumentParser()
ap.add_argument("--ckpts", nargs="+", required=True)
ap.add_argument("--weights", nargs="*", type=float, default=None)
ap.add_argument("--dst", required=True)
a = ap.parse_args()

w = a.weights or [1.0 / len(a.ckpts)] * len(a.ckpts)
assert len(w) == len(a.ckpts)
s = sum(w)
w = [x / s for x in w]
print("soup weights:", dict(zip(a.ckpts, w)))

base = Gemma3ForConditionalGeneration.from_pretrained(a.ckpts[0], dtype=torch.float32)
sd = base.state_dict()
for k in sd:
    sd[k] = sd[k].to(torch.float32) * w[0]

for path, wi in zip(a.ckpts[1:], w[1:]):
    m = Gemma3ForConditionalGeneration.from_pretrained(path, dtype=torch.float32)
    other = m.state_dict()
    assert set(other.keys()) == set(sd.keys()), "state dict mismatch"
    for k in sd:
        sd[k] += other[k].to(torch.float32) * wi
    del m, other

base.load_state_dict(sd)
base = base.to(torch.bfloat16)
# HF refuses to serialise do_sample=False alongside temperature/top_k; write a
# valid config here and put the greedy values into the json below, which is what
# vLLM actually reads.
base.generation_config.do_sample = True
base.generation_config.temperature = None
base.generation_config.top_k = None
base.generation_config.top_p = None
os.makedirs(a.dst, exist_ok=True)
base.save_pretrained(a.dst, safe_serialization=True)
AutoTokenizer.from_pretrained(a.ckpts[0]).save_pretrained(a.dst)
for f in ("preprocessor_config.json", "processor_config.json", "added_tokens.json"):
    p = os.path.join(a.ckpts[0], f)
    if os.path.exists(p):
        shutil.copy(p, os.path.join(a.dst, f))
gc_path = os.path.join(a.dst, "generation_config.json")
gc = json.load(open(gc_path)) if os.path.exists(gc_path) else {}
gc.update({"do_sample": False, "temperature": 0.0, "top_k": 1, "top_p": 1.0,
           "bos_token_id": 2, "eos_token_id": [1, 106], "pad_token_id": 0})
json.dump(gc, open(gc_path, "w"), indent=2)
print("wrote", a.dst)
