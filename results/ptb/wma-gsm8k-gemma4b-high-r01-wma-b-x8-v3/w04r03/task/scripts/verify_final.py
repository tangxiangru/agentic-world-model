#!/usr/bin/env python3
"""Pre-deadline check on final_model/: the grader loads it with vLLM from a fresh process.

Checks the things that have silently cost whole runs: the directory is a full model (not
an adapter), evaluate.py's model_type() resolves it to the gemma template, the tokenizer
and its chat template are present, generation_config keeps <end_of_turn> as an eos and
asks for greedy decoding, and transformers can actually instantiate the weights.
"""
from __future__ import annotations

import json
import os
import sys

os.environ.setdefault("HF_HOME", "/home/ben/hf_cache")

import torch

D = sys.argv[1] if len(sys.argv) > 1 else "/home/ben/task/final_model"
ok = True


def chk(cond: bool, msg: str) -> None:
    global ok
    print(("PASS  " if cond else "FAIL  ") + msg)
    ok = ok and bool(cond)


files = set(os.listdir(D))
chk("config.json" in files, "config.json present")
chk(not any(f.startswith("adapter_") for f in files), "not a LoRA adapter directory")
chk(any(f.endswith(".safetensors") for f in files), "safetensors weights present")
chk("tokenizer.json" in files, "tokenizer.json present")
chk("generation_config.json" in files, "generation_config.json present")

cfg = json.load(open(os.path.join(D, "config.json")))
arch = cfg["architectures"][0]
chk("gemma" in arch.lower(), f"architectures[0]={arch} -> evaluate.py picks gemma3.jinja")
dt = cfg.get("dtype") or cfg.get("torch_dtype")
chk(dt == "bfloat16", f"weights dtype {dt} (must be bf16, not fp32, for --gpu-memory-utilization 0.3)")

gc = json.load(open(os.path.join(D, "generation_config.json")))
chk(106 in (gc.get("eos_token_id") or []), f"eos_token_id {gc.get('eos_token_id')} contains 106 <end_of_turn>")
chk(gc.get("temperature") == 0.0, f"temperature {gc.get('temperature')} -> vLLM decodes greedily")

sz = sum(os.path.getsize(os.path.join(D, f)) for f in files if f.endswith(".safetensors"))
chk(sz > 7e9, f"weights {sz/1e9:.1f} GB")

from transformers import AutoModelForCausalLM, AutoTokenizer

tok = AutoTokenizer.from_pretrained(D)
tmpl = open("/home/ben/task/templates/gemma3.jinja").read()
rendered = tok.apply_chat_template([{"role": "user", "content": "hi"}], tokenize=False, add_generation_prompt=True)
print("     rendered with the shipped template:", repr(rendered))
tok.chat_template = tmpl
rendered_grader = tok.apply_chat_template([{"role": "user", "content": "hi"}], tokenize=False, add_generation_prompt=True)
chk(rendered == rendered_grader, "shipped chat template renders identically to templates/gemma3.jinja")

m = AutoModelForCausalLM.from_pretrained(D, dtype=torch.bfloat16)
out = m(input_ids=torch.tensor([[2, 105, 2364, 108, 3689]]))
chk(torch.isfinite(out.logits).all().item(), "forward pass on CPU produces finite logits")
print("\n" + ("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED"))
sys.exit(0 if ok else 1)
