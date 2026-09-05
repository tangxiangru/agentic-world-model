#!/usr/bin/env python3
"""Pre-deadline check on final_model/: the grader loads it with vLLM from a fresh
process, so verify here that (a) transformers can load it from disk, (b) the
tokenizer and the multimodal processor files are present, (c) the decode config
is the frozen greedy one with <end_of_turn> among the stop ids, and (d) the
model actually emits <end_of_turn> on a real graded prompt.
"""
import json
import os
import sys

import torch
from transformers import AutoConfig, AutoProcessor, AutoTokenizer, Gemma3ForConditionalGeneration

D = sys.argv[1] if len(sys.argv) > 1 else "final_model"
ok = True


def check(name, cond, detail=""):
    global ok
    print(f"[{'ok ' if cond else 'FAIL'}] {name} {detail}")
    ok = ok and bool(cond)


need = ["config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json",
        "special_tokens_map.json", "preprocessor_config.json", "processor_config.json",
        "model.safetensors.index.json"]
for f in need:
    check(f"file {f}", os.path.exists(os.path.join(D, f)))

cfg = AutoConfig.from_pretrained(D)
check("architecture", "Gemma3" in cfg.architectures[0], cfg.architectures[0])
gen = json.load(open(os.path.join(D, "generation_config.json")))
check("greedy temperature 0.0", gen.get("temperature") == 0.0, str(gen.get("temperature")))
check("no top_k/top_p", "top_k" not in gen and "top_p" not in gen)
check("eos ids include 106", 106 in (gen.get("eos_token_id") or []), str(gen.get("eos_token_id")))

tok = AutoTokenizer.from_pretrained(D)
check("tokenizer loads", tok.convert_tokens_to_ids("<end_of_turn>") == 106)
try:
    AutoProcessor.from_pretrained(D)
    check("processor loads", True)
except Exception as e:  # noqa: BLE001
    check("processor loads", False, str(e)[:120])

model = Gemma3ForConditionalGeneration.from_pretrained(D, dtype=torch.bfloat16, device_map="cuda")
tmpl = open("templates/gemma3.jinja").read()
q = ("A robe takes 2 bolts of blue fiber and half that much white fiber. "
     "How many bolts in total does it take?")
from local_eval import MATH_PROMPT_TEMPLATE, fewshot_system, last_number_normalized
msgs = [{"role": "system", "content": fewshot_system()},
        {"role": "user", "content": MATH_PROMPT_TEMPLATE.format(prompt=q)}]
text = tok.apply_chat_template(msgs, chat_template=tmpl, tokenize=False, add_generation_prompt=True)
ids = tok(text, add_special_tokens=False, return_tensors="pt").to(model.device)
out = model.generate(**ids, max_new_tokens=512, do_sample=False, eos_token_id=[1, 106])
gen_ids = out[0][ids["input_ids"].shape[1]:]
completion = tok.decode(gen_ids, skip_special_tokens=True)
check("emits <end_of_turn>", int(gen_ids[-1]) in (1, 106), f"last id {int(gen_ids[-1])}")
check("graded answer is 3", last_number_normalized(completion) == "3", repr(completion[-90:]))
print("\n--- completion ---\n" + completion)
print("\nALL OK" if ok else "\nSOMETHING FAILED")
sys.exit(0 if ok else 1)
