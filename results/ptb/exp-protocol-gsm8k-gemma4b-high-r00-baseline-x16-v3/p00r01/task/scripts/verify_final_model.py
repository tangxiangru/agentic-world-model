#!/usr/bin/env python3
"""Pre-deadline check on final_model/: everything the grader needs is present and
the weights actually load in a fresh process.

Covers pitfalls.yaml final_model_not_loadable and template_unreachable.
"""
from __future__ import annotations

import json
import os
import sys

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

D = sys.argv[1] if len(sys.argv) > 1 else "/home/ben/task/final_model"
TEMPLATE = "/home/ben/task/templates/gemma3.jinja"

fail = []

required = [
    "config.json", "generation_config.json", "tokenizer.json",
    "tokenizer_config.json", "special_tokens_map.json",
    "preprocessor_config.json", "processor_config.json",
    "model.safetensors.index.json",
]
for f in required:
    if not os.path.exists(os.path.join(D, f)):
        fail.append(f"missing {f}")

cfg = AutoConfig.from_pretrained(D)
arch = cfg.architectures[0]
print("architectures:", cfg.architectures)
# evaluate.py picks the chat template off this string when the path has no
# model name in it
if "gemma" not in arch.lower():
    fail.append(f"evaluate.py model_type() would not resolve {arch} to gemma3.jinja")

gen = json.load(open(os.path.join(D, "generation_config.json")))
print("eos_token_id:", gen.get("eos_token_id"))
tok = AutoTokenizer.from_pretrained(D)
eot = tok.convert_tokens_to_ids("<end_of_turn>")
eos_ids = gen.get("eos_token_id")
eos_ids = eos_ids if isinstance(eos_ids, list) else [eos_ids]
if eot not in eos_ids:
    fail.append(f"<end_of_turn> ({eot}) not in generation_config eos_token_id {eos_ids}")

# the grader renders with templates/gemma3.jinja, not whatever the checkpoint ships
tok.chat_template = open(TEMPLATE).read()
rendered = tok.apply_chat_template(
    [{"role": "user", "content": "What is 2+2?"}],
    tokenize=False, add_generation_prompt=True,
)
ids = tok(rendered, add_special_tokens=False)["input_ids"]
if ids[0] != tok.bos_token_id or ids[1] == tok.bos_token_id:
    fail.append(f"template/bos problem: first ids {ids[:3]}")
print("rendered prompt ok, first ids:", ids[:4])

print("loading weights on CPU...")
m = AutoModelForCausalLM.from_pretrained(D, dtype=torch.bfloat16)
n = sum(p.numel() for p in m.parameters())
print(f"loaded {type(m).__name__} {n/1e9:.2f}B params")
if n < 3e9:
    fail.append(f"only {n/1e9:.2f}B params loaded")

if fail:
    print("\nFAILED:")
    for f in fail:
        print("  -", f)
    sys.exit(1)
print("\nfinal_model OK")
