"""Reproduce the exact prompt string the grader sends to vLLM."""
import json, os
from inspect_ai.dataset import hf_dataset
from inspect_evals.gsm8k.gsm8k import record_to_sample, sample_to_fewshot, MATH_PROMPT_TEMPLATE
from transformers import AutoTokenizer

SNAP = "/home/ben/hf_cache/hub/models--google--gemma-3-4b-pt/snapshots/cc012e0a6d0787b4adcc0fa2c4da74402494554d"

fewshots = hf_dataset(path="openai/gsm8k", data_dir="main", split="train",
                      sample_fields=record_to_sample, shuffle=True, seed=42, limit=10)
sysmsg = "\n\n".join([sample_to_fewshot(s) for s in fewshots])
open("data/eval_system_message.txt","w").write(sysmsg)
print("SYSTEM MESSAGE chars:", len(sysmsg))
print(sysmsg[:1200])
print("......")
print(sysmsg[-600:])

tok = AutoTokenizer.from_pretrained(SNAP)
tpl = open("templates/gemma3.jinja").read()
tok.chat_template = tpl

q = "Janet has 3 apples and buys 5 more. How many does she have?"
user = MATH_PROMPT_TEMPLATE.replace("{prompt}", q)
msgs = [{"role":"system","content":sysmsg},{"role":"user","content":user}]
rendered = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
print("\n===== RENDERED (tail 900) =====")
print(rendered[-900:])
ids = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True)
print("\nPROMPT TOKENS (10-shot + q):", len(ids))
print("SYS-only tokens:", len(tok(sysmsg)['input_ids']))
print("eos id:", tok.eos_token_id, "end_of_turn id:", tok.convert_tokens_to_ids("<end_of_turn>"), "bos:", tok.bos_token_id)
print("repr first 200 rendered:", repr(rendered[:200]))
open("data/example_eval_prompt.txt","w").write(rendered)
