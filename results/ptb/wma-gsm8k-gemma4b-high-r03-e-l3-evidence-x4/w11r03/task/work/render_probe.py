import json, os
from transformers import AutoTokenizer
from inspect_ai.dataset import hf_dataset
from inspect_evals.gsm8k.gsm8k import record_to_sample, sample_to_fewshot, MATH_PROMPT_TEMPLATE

P='/home/ben/hf_cache/hub/models--google--gemma-3-4b-pt/snapshots/cc012e0a6d0787b4adcc0fa2c4da74402494554d'
tok=AutoTokenizer.from_pretrained(P)
tpl=open('templates/gemma3.jinja').read()

fewshots = hf_dataset(path="openai/gsm8k", data_dir="main", split="train",
                      sample_fields=record_to_sample, shuffle=True, seed=42, limit=10)
sysmsg = "\n\n".join([sample_to_fewshot(s) for s in fewshots])
open('work/fewshot_system.txt','w').write(sysmsg)
print("SYSTEM MSG tokens:", len(tok(sysmsg).input_ids))
print("=== first 900 chars of system ===")
print(sysmsg[:900])
print("...")
q = "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?"
user = MATH_PROMPT_TEMPLATE.format(prompt=q)
msgs=[{"role":"system","content":sysmsg},{"role":"user","content":user}]
rendered = tok.apply_chat_template(msgs, chat_template=tpl, tokenize=False, add_generation_prompt=True)
print("=== RENDERED (tail 1200) ===")
print(rendered[-1200:])
print("=== full prompt tokens:", len(tok(rendered, add_special_tokens=False).input_ids))
msgs0=[{"role":"user","content":user}]
r0 = tok.apply_chat_template(msgs0, chat_template=tpl, tokenize=False, add_generation_prompt=True)
print("=== ZERO-SHOT RENDER ===")
print(repr(r0))
print("zeroshot tokens:", len(tok(r0, add_special_tokens=False).input_ids))
