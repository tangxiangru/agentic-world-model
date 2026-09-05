#!/usr/bin/env python3
"""Uniform weight average (model soup) of two checkpoints fine-tuned from the
same base. Writes a full loadable model directory with a greedy generation_config.
"""
import argparse, json, os, shutil
import torch
from transformers import AutoProcessor, AutoTokenizer, Gemma3ForConditionalGeneration

ap = argparse.ArgumentParser()
ap.add_argument("--a", required=True)
ap.add_argument("--b", required=True)
ap.add_argument("--wa", type=float, default=0.5)
ap.add_argument("--out", required=True)
args = ap.parse_args()

ma = Gemma3ForConditionalGeneration.from_pretrained(args.a, dtype=torch.bfloat16)
sb = Gemma3ForConditionalGeneration.from_pretrained(args.b, dtype=torch.bfloat16).state_dict()
sa = ma.state_dict()
assert set(sa) == set(sb), (len(sa), len(sb))
with torch.no_grad():
    for k in sa:
        sa[k].mul_(args.wa).add_(sb[k].to(sa[k].dtype), alpha=1.0 - args.wa)
ma.load_state_dict(sa)
ma.save_pretrained(args.out)
# transformers refuses to save temperature=0 alongside do_sample=False, so the
# greedy decode config is written straight to disk after save_pretrained.
gcp = os.path.join(args.out, "generation_config.json")
g = json.load(open(gcp))
g.pop("top_k", None); g.pop("top_p", None)
g["do_sample"] = False; g["temperature"] = 0.0
json.dump(g, open(gcp, "w"), indent=2)
AutoProcessor.from_pretrained(args.a).save_pretrained(args.out)
AutoTokenizer.from_pretrained(args.a).save_pretrained(args.out)
gc = json.load(open(os.path.join(args.out, "generation_config.json")))
print(json.dumps(gc))
print("saved", args.out)
