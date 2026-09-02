#!/usr/bin/env python3
"""Sanity check: final_model loads and generates with stock transformers."""
import os
os.environ.setdefault("HF_HOME", "/home/ben/hf_cache")
os.environ.setdefault("HF_HUB_CACHE", "/home/ben/hf_cache/hub")
import torch
from transformers import AutoTokenizer, Gemma3ForConditionalGeneration

PROMPT = """
Solve the following math problem step by step. The last line of your response should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.

{prompt}

Remember to put your answer on its own line at the end in the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem, and you do not need to use a \\boxed command.

Reasoning:
""".strip()

Q = ("A bakery sells muffins in boxes of 6. On Monday they sold 14 boxes and on Tuesday "
     "they sold 9 boxes. Each muffin costs $3. How much money did they make in total?")

tok = AutoTokenizer.from_pretrained("final_model")
tok.chat_template = open("templates/gemma3.jinja").read()
model = Gemma3ForConditionalGeneration.from_pretrained(
    "final_model", dtype=torch.bfloat16, device_map="cuda")
text = tok.apply_chat_template([{"role": "user", "content": PROMPT.format(prompt=Q)}],
                               tokenize=False, add_generation_prompt=True)
ids = tok(text, add_special_tokens=False, return_tensors="pt").to("cuda")
out = model.generate(**ids, max_new_tokens=320, do_sample=False)
print(tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True))
print("--- arch:", model.config.architectures, "dtype:", model.dtype)
