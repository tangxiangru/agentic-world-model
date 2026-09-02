#!/usr/bin/env python3
"""Convert a training checkpoint into a bf16, greedy-decoding, ready-to-serve dir."""
import json, os, shutil, sys
import torch
from transformers import AutoTokenizer, Gemma3ForCausalLM

src, dst = sys.argv[1], sys.argv[2]
tok_src = sys.argv[3] if len(sys.argv) > 3 else "runs/sft1_bf16"

m = Gemma3ForCausalLM.from_pretrained(src, dtype=torch.bfloat16)
m.config.dtype = "bfloat16"
os.makedirs(dst, exist_ok=True)
m.save_pretrained(dst, safe_serialization=True)
tok = AutoTokenizer.from_pretrained(tok_src)
tok.save_pretrained(dst)
for junk in ("chat_template.jinja",):
    p = os.path.join(dst, junk)
    if os.path.exists(p):
        os.remove(p)
cfg = {"bos_token_id": 2, "eos_token_id": [1, 106], "pad_token_id": 0,
       "cache_implementation": "hybrid", "do_sample": False, "temperature": 0.0,
       "top_p": 1.0, "top_k": -1, "transformers_version": "4.57.3"}
json.dump(cfg, open(os.path.join(dst, "generation_config.json"), "w"), indent=2)
print("prepared", dst)
