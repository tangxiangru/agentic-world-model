#!/usr/bin/env python3
"""Uniform weight average (model soup) of several checkpoints."""
import sys, torch, shutil, os, json
from transformers import Gemma3ForCausalLM

dst = sys.argv[1]
srcs = sys.argv[2:]
sd = None
for i, s in enumerate(srcs):
    m = Gemma3ForCausalLM.from_pretrained(s, dtype=torch.float32)
    cur = m.state_dict()
    if sd is None:
        sd = {k: v.clone() for k, v in cur.items()}
    else:
        for k in sd:
            sd[k] += cur[k]
    del m, cur
    print("added", s, flush=True)
for k in sd:
    sd[k] = (sd[k] / len(srcs)).to(torch.bfloat16)
model = Gemma3ForCausalLM.from_pretrained(srcs[0], dtype=torch.bfloat16)
model.load_state_dict(sd)
model.save_pretrained(dst, safe_serialization=True)
for f in ["tokenizer.json", "tokenizer.model", "tokenizer_config.json",
          "special_tokens_map.json", "added_tokens.json", "chat_template.jinja"]:
    p = os.path.join(srcs[0], f)
    if os.path.exists(p):
        shutil.copy(p, os.path.join(dst, f))
print("saved", dst)
