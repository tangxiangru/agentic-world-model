#!/usr/bin/env python3
"""Linearly interpolate the weights of two checkpoints (WiSE-FT style soup)."""
import json, os, sys
import torch
from transformers import AutoTokenizer, Gemma3ForCausalLM

a_path, b_path, alpha, dst = sys.argv[1], sys.argv[2], float(sys.argv[3]), sys.argv[4]
from transformers import GenerationConfig
a = Gemma3ForCausalLM.from_pretrained(a_path, dtype=torch.float32)
a.generation_config = GenerationConfig(bos_token_id=2, eos_token_id=[1, 106], pad_token_id=0)
b = Gemma3ForCausalLM.from_pretrained(b_path, dtype=torch.float32)
sa, sb = a.state_dict(), b.state_dict()
assert set(sa) == set(sb)
with torch.no_grad():
    for k in sa:
        sa[k].mul_(alpha).add_(sb[k], alpha=1.0 - alpha)
a.load_state_dict(sa)
a = a.to(torch.bfloat16)
a.config.dtype = "bfloat16"
os.makedirs(dst, exist_ok=True)
a.save_pretrained(dst, safe_serialization=True)
AutoTokenizer.from_pretrained("runs/sft1_bf16").save_pretrained(dst)
p = os.path.join(dst, "chat_template.jinja")
if os.path.exists(p):
    os.remove(p)
json.dump({"bos_token_id": 2, "eos_token_id": [1, 106], "pad_token_id": 0,
           "cache_implementation": "hybrid", "do_sample": False, "temperature": 0.0,
           "top_p": 1.0, "top_k": -1, "transformers_version": "4.57.3"},
          open(os.path.join(dst, "generation_config.json"), "w"), indent=2)
print("soup saved", dst, "alpha", alpha)
