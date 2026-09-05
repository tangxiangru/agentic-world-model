#!/usr/bin/env python3
"""P(<end_of_turn>) vs P(newline) right after the correct answer, IN the 10-shot context."""
import sys, json, torch
from transformers import AutoTokenizer, Gemma3ForConditionalGeneration

model_path = sys.argv[1]
log_path = sys.argv[2]
template = open("templates/gemma3.jinja").read()
tok = AutoTokenizer.from_pretrained(model_path)
eot = tok.convert_tokens_to_ids("<end_of_turn>")

d = json.load(open(log_path))
s = d["samples"][0]
chat = [{"role": m["role"], "content": m["content"]} for m in s["messages"] if m["role"] in ("system", "user")]
pid = tok.apply_chat_template(chat, chat_template=template, add_generation_prompt=True, tokenize=True)
partial = "Cody eats 5 x 3 = <<5*3=15>>15 cookies.\nTogether, they eat 5 + 15 = <<5+15=20>>20 cookies.\n\nANSWER: 20"
cid = tok(partial, add_special_tokens=False).input_ids
ids = pid + cid

model = Gemma3ForConditionalGeneration.from_pretrained(model_path, torch_dtype=torch.bfloat16,
                                                       attn_implementation="eager").cuda().eval()
with torch.no_grad():
    logits = model(torch.tensor([ids]).cuda()).logits[0, -1]
    probs = torch.softmax(logits.float(), dim=-1)
    top = torch.topk(probs, 3)
top_str = ", ".join(f"{repr(tok.decode([t]))}={p:.3f}" for p, t in zip(top.values.tolist(), top.indices.tolist()))
print(f"{model_path}: P(eot)={probs[eot].item():.4f} | top3: {top_str}")
