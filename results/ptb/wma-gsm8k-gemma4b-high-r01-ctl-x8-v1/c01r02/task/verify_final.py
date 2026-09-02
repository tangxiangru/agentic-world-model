#!/usr/bin/env python3
"""Pre-deadline check on final_model/ (pitfall final_model_not_loadable).

The grader loads final_model/ with vLLM from a fresh process and renders prompts with
templates/gemma3.jinja. This checks, on CPU only, that:
  * evaluate.py's model_type() resolves the directory to the gemma template
  * config/tokenizer/weights load with transformers from a cold start
  * the tokenizer round-trips <end_of_turn> to id 106 and generation_config lists it as eos
  * generation_config.json still asks for greedy decoding (exp-03: +16 points)
"""
from __future__ import annotations

import json
import os
import sys

os.environ.setdefault("HF_HOME", "/home/ben/hf_cache")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

path = sys.argv[1] if len(sys.argv) > 1 else "/home/ben/task/final_model"
ok = True


def check(name, cond, detail=""):
    global ok
    ok = ok and bool(cond)
    print(f"{'PASS' if cond else 'FAIL'}  {name}{(' - ' + detail) if detail else ''}")


cfg = json.load(open(os.path.join(path, "config.json")))
arch = cfg["architectures"][0].lower()
check("architecture routes to templates/gemma3.jinja", "gemma" in arch, arch)

gc = json.load(open(os.path.join(path, "generation_config.json")))
check("generation_config is greedy", gc.get("temperature") == 0.0 and gc.get("do_sample") is False, json.dumps(gc))
check("end_of_turn (106) is an eos id", 106 in (gc.get("eos_token_id") or []), str(gc.get("eos_token_id")))
check("no top_k/top_p left to re-enable sampling", "top_k" not in gc and "top_p" not in gc)

from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer  # noqa: E402

tok = AutoTokenizer.from_pretrained(path)
ids = tok("<end_of_turn>", add_special_tokens=False)["input_ids"]
check("tokenizer encodes <end_of_turn> to a single id 106", ids == [106], str(ids))

AutoConfig.from_pretrained(path)
import torch  # noqa: E402

m = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16)
n = sum(p.numel() for p in m.parameters())
check("weights load on CPU with transformers", n > 4e9, f"{n/1e9:.2f}B parameters")

# render one grading-shaped prompt exactly as the harness will
tok.chat_template = open("/home/ben/task/templates/gemma3.jinja").read()
msgs = [{"role": "system", "content": open("/home/ben/task/data/eval_system_message.txt").read()},
        {"role": "user", "content": "Solve the following math problem step by step.\n\n2+2?\n\nReasoning:"}]
r = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
check("grader-shaped prompt renders", r.startswith("<bos><start_of_turn>user") and r.endswith("<start_of_turn>model\n"))

sizes = {f: os.path.getsize(os.path.join(path, f)) for f in os.listdir(path)}
check("safetensors present", any(f.endswith(".safetensors") for f in sizes), f"{sum(sizes.values())/2**30:.1f} GiB total")
print(json.dumps(sorted(sizes), indent=1))
print("ALL OK" if ok else "SOMETHING FAILED")
sys.exit(0 if ok else 1)
