#!/usr/bin/env python3
"""Reproduce the real 10-shot eval prompt from the log and greedy-generate."""
import sys, json, torch
from transformers import AutoTokenizer, Gemma3ForConditionalGeneration

model_path = sys.argv[1]
log_path = sys.argv[2]
template = open("templates/gemma3.jinja").read()
tok = AutoTokenizer.from_pretrained(model_path)
eot = tok.convert_tokens_to_ids("<end_of_turn>")

d = json.load(open(log_path))
s = d["samples"][0]
msgs = s["messages"]
chat = [{"role": m["role"], "content": m["content"]} for m in msgs if m["role"] in ("system", "user")]
pid = tok.apply_chat_template(chat, chat_template=template, add_generation_prompt=True, tokenize=True)
print(f"prompt tokens: {len(pid)}")

model = Gemma3ForConditionalGeneration.from_pretrained(model_path, torch_dtype=torch.bfloat16,
                                                       attn_implementation="eager").cuda().eval()
gen = list(pid)
with torch.no_grad():
    for _ in range(400):
        o = model(torch.tensor([gen]).cuda())
        nt = int(o.logits[0, -1].argmax())
        gen.append(nt)
        if nt == eot or nt == 1:
            break
text = tok.decode(gen[len(pid):])
print(f"=== {model_path} ===")
print(f"stopped_at_eot={gen[-1]==eot} n_new_tokens={len(gen)-len(pid)}")
print(repr(text[:800]))
