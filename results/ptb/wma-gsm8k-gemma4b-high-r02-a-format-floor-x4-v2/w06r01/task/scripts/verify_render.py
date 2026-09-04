#!/usr/bin/env python3
"""Precondition 1 of the exp-02 verdict: prove that a training row carrying a
K=10 few-shot prefix renders byte-for-byte identically to the prompt the grader
actually sends, through the same templates/gemma3.jinja.
"""
import json, sys
sys.path.insert(0, "/home/ben/task/scripts")
from transformers import AutoTokenizer
from train_sft import SNAPSHOT, TEMPLATE, fewshot_prefix
from inspect_evals.gsm8k.gsm8k import MATH_PROMPT_TEMPLATE
import datasets

ev = json.load(open("/home/ben/task/data/eval_prompt.json"))
eval_sys = ev["fewshot_system"]

# 1. our few-shot builder must reproduce the harness's exemplar block exactly.
#    Rebuild the harness's own 10 demos (train split, shuffle seed 42, limit 10)
#    and run them through fewshot_prefix().
from inspect_ai.dataset import hf_dataset
from inspect_evals.gsm8k.gsm8k import record_to_sample
fs = hf_dataset(path="openai/gsm8k", data_dir="main", split="train",
                sample_fields=record_to_sample, shuffle=True, seed=42, limit=10)
demos = [{"q": s.input, "r": s.metadata["reasoning"], "a": s.target} for s in fs]
ours = fewshot_prefix(demos)
assert ours == eval_sys, "few-shot exemplar block differs from the harness's"
print("PASS  few-shot exemplar block is byte-identical to the harness's system message")

# 2. the full rendered prompt must match too.
tok = AutoTokenizer.from_pretrained(SNAPSHOT)
tok.chat_template = open(TEMPLATE).read()
q = "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?"
user = MATH_PROMPT_TEMPLATE.replace("{prompt}", q)
msgs = [{"role": "system", "content": ours}, {"role": "user", "content": user}]
train_render = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
eval_render = tok.apply_chat_template(
    [{"role": "system", "content": eval_sys}, {"role": "user", "content": user}],
    tokenize=False, add_generation_prompt=True)
assert train_render == eval_render
print("PASS  full K=10 render matches, %d tokens" %
      len(tok(train_render, add_special_tokens=False)["input_ids"]))
tgt = "Natalia sold 48/2 = 24 clips in May.\nNatalia sold 48+24 = 72 clips altogether.\n\nANSWER: 72<end_of_turn>"
n = len(tok(train_render + tgt, add_special_tokens=False)["input_ids"])
print("PASS  worst-case K=10 training row = %d tokens (block must exceed this)" % n)
print(train_render[-400:] + tgt)
