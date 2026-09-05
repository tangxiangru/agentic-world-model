#!/usr/bin/env python3
"""Uniform weight average ('model soup') of two or more checkpoints that share
an architecture.  Averaging is done in float32 and cast back to bfloat16.
The output carries the same tokenizer and the same greedy generation_config.
"""
import json, os, shutil, sys
import torch
from transformers import AutoTokenizer, Gemma3ForConditionalGeneration

out = sys.argv[1]
srcs = sys.argv[2:]
assert len(srcs) >= 2, "need at least two checkpoints"

acc = None
for i, s in enumerate(srcs):
    m = Gemma3ForConditionalGeneration.from_pretrained(s, dtype=torch.float32,
                                                       device_map="cpu")
    sd = m.state_dict()
    if acc is None:
        acc = {k: v.clone() for k, v in sd.items()}
    else:
        for k in acc:
            acc[k] += sd[k]
    del m, sd
    print(f"  added {s}", flush=True)

for k in acc:
    acc[k] /= len(srcs)
    acc[k] = acc[k].to(torch.bfloat16)

model = Gemma3ForConditionalGeneration.from_pretrained(srcs[0],
                                                       dtype=torch.bfloat16,
                                                       device_map="cpu")
model.load_state_dict(acc)
for f in ("temperature", "top_k", "top_p"):
    if getattr(model.generation_config, f, None) is not None:
        setattr(model.generation_config, f, None)
model.generation_config.do_sample = False
os.makedirs(out, exist_ok=True)
model.save_pretrained(out)
AutoTokenizer.from_pretrained(srcs[0]).save_pretrained(out)
for f in ("preprocessor_config.json", "processor_config.json"):
    p = os.path.join(srcs[0], f)
    if os.path.exists(p):
        shutil.copy(os.path.realpath(p), os.path.join(out, f))
json.dump({"bos_token_id": 2, "eos_token_id": [1, 106], "pad_token_id": 0,
           "cache_implementation": "hybrid", "do_sample": False,
           "temperature": 0.0},
          open(os.path.join(out, "generation_config.json"), "w"), indent=2)
print("soup saved to", out, "from", len(srcs), "checkpoints")
