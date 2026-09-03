#!/usr/bin/env python3
"""Uniformly average the weights of several checkpoints of the same architecture."""
import argparse, os, shutil, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--models", nargs="+", required=True)
ap.add_argument("--weights", nargs="*", type=float, default=None)
ap.add_argument("--out", required=True)
a = ap.parse_args()

w = a.weights or [1.0 / len(a.models)] * len(a.models)
assert len(w) == len(a.models)
s = sum(w); w = [x / s for x in w]
print("weights", w)

base = AutoModelForCausalLM.from_pretrained(a.models[0], dtype=torch.float32)
sd = base.state_dict()
for k in sd:
    sd[k] = sd[k] * w[0]
for m, wi in zip(a.models[1:], w[1:]):
    other = AutoModelForCausalLM.from_pretrained(m, dtype=torch.float32)
    osd = other.state_dict()
    for k in sd:
        sd[k] += osd[k] * wi
    del other, osd
base.load_state_dict(sd)
base = base.to(torch.bfloat16)
gc = base.generation_config
gc.do_sample = True; gc.temperature = 1.0; gc.top_p = 0.95; gc.top_k = 64
base.save_pretrained(a.out)
tok = AutoTokenizer.from_pretrained(a.models[0])
tok.save_pretrained(a.out)
for f in ["preprocessor_config.json", "processor_config.json", "generation_config.json"]:
    src = os.path.join(a.models[0], f)
    if os.path.exists(src):
        shutil.copy(src, os.path.join(a.out, f))
print("saved", a.out)
