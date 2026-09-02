"""Uniform weight average of two or more checkpoints (model soup)."""
from __future__ import annotations
import argparse, torch
from transformers import Gemma3ForConditionalGeneration

ap = argparse.ArgumentParser()
ap.add_argument("--src", nargs="+", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()

acc = None
for i, s in enumerate(a.src):
    print("[soup] loading", s, flush=True)
    m = Gemma3ForConditionalGeneration.from_pretrained(s, dtype=torch.float32)
    sd = m.state_dict()
    if acc is None:
        acc = {k: v.clone() for k, v in sd.items()}
    else:
        for k in acc:
            acc[k] += sd[k]
    del m, sd
for k in acc:
    acc[k] /= len(a.src)
print("[soup] rebuilding", flush=True)
m = Gemma3ForConditionalGeneration.from_pretrained(a.src[0], dtype=torch.float32)
m.load_state_dict(acc)
m = m.to(torch.bfloat16)
m.save_pretrained(a.out, safe_serialization=True)
print("[soup] saved", a.out, flush=True)
