#!/usr/bin/env python3
"""Uniform weight average of two checkpoints of the same architecture.

exp-02/final and exp-03/final are consecutive points on one training
trajectory (exp-03 continued exp-02), so their weights live in the same basin
and a plain average is well defined.
"""
import argparse, json, os, shutil
import torch
from transformers import AutoTokenizer, Gemma3ForConditionalGeneration

ap = argparse.ArgumentParser()
ap.add_argument("--a", required=True)
ap.add_argument("--b", required=True)
ap.add_argument("--alpha", type=float, default=0.5, help="weight on --a")
ap.add_argument("--out", required=True)
args = ap.parse_args()

ma = Gemma3ForConditionalGeneration.from_pretrained(args.a, dtype=torch.bfloat16)
mb = Gemma3ForConditionalGeneration.from_pretrained(args.b, dtype=torch.bfloat16)
sa, sb = ma.state_dict(), mb.state_dict()
assert set(sa) == set(sb), "state dicts differ"
# Gemma3 ties lm_head.weight to embed_tokens.weight, so those two keys share one
# storage. Averaging in place without deduplicating would hit that tensor twice
# and leave it at 0.25a+0.75b instead of the uniform 0.5 the card claims.
n_avg, n_tied, seen = 0, 0, set()
for k in sa:
    if not sa[k].is_floating_point():
        continue
    ptr = sa[k].data_ptr()
    if ptr in seen:
        n_tied += 1
        continue
    seen.add(ptr)
    sa[k].mul_(args.alpha).add_(sb[k].to(sa[k].dtype), alpha=1 - args.alpha)
    n_avg += 1
ma.load_state_dict(sa)
print(f"averaged {n_avg}/{len(sa)} float tensors at alpha={args.alpha}; "
      f"{n_tied} skipped as tied duplicates")

# exp-02/final carries the greedy generation_config (do_sample False +
# temperature 0.0) that vLLM needs; transformers refuses to save that pair, so
# save_pretrained below would die after both models were loaded. Reset it here.
from transformers import GenerationConfig
ma.generation_config = GenerationConfig(
    bos_token_id=2, eos_token_id=[1, 106], pad_token_id=0,
    cache_implementation="hybrid")

os.makedirs(args.out, exist_ok=True)
ma.save_pretrained(args.out, safe_serialization=True)
AutoTokenizer.from_pretrained(args.a).save_pretrained(args.out)
for f in ["preprocessor_config.json", "processor_config.json"]:
    p = os.path.join(args.a, f)
    if os.path.exists(p):
        shutil.copy(p, os.path.join(args.out, f))
json.dump({"bos_token_id": 2, "eos_token_id": [1, 106], "pad_token_id": 0,
           "cache_implementation": "hybrid", "do_sample": False,
           "temperature": 0.0, "transformers_version": "4.57.3"},
          open(os.path.join(args.out, "generation_config.json"), "w"), indent=2)
print("saved", args.out)
