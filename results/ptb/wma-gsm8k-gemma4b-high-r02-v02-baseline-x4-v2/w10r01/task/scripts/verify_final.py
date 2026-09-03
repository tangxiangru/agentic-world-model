#!/usr/bin/env python3
"""Pre-submission check on final_model/.

The grader loads final_model/ with vLLM from a fresh process, so this asserts
the things that silently turn a good checkpoint into a base-model score:
files present, config loadable, greedy generation_config intact, weights bf16
and not the base snapshot's, and one real generation that stops on
<end_of_turn> and ends with an ANSWER line.
"""
from __future__ import annotations

import json
import os
import sys

import torch
from safetensors import safe_open
from transformers import AutoConfig, AutoProcessor, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import SNAPSHOT, render_prompt  # noqa: E402

FINAL = sys.argv[1] if len(sys.argv) > 1 else "/home/ben/task/final_model"
REQUIRED = [
    "config.json",
    "generation_config.json",
    "model.safetensors.index.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "preprocessor_config.json",
    "processor_config.json",
]

ok = True


def check(name, cond, detail=""):
    global ok
    ok = ok and bool(cond)
    print(f"{'PASS' if cond else 'FAIL'}  {name}{(': ' + detail) if detail else ''}")


for f in REQUIRED:
    check(f"file {f}", os.path.exists(os.path.join(FINAL, f)))

cfg = AutoConfig.from_pretrained(FINAL)
check("architecture", cfg.architectures == ["Gemma3ForConditionalGeneration"], str(cfg.architectures))
check("dtype bfloat16", str(getattr(cfg, "dtype", None)).endswith("bfloat16"), str(getattr(cfg, "dtype", None)))
check("evaluate.py model_type -> gemma", "gemma" in cfg.architectures[0].lower())

gen = json.load(open(os.path.join(FINAL, "generation_config.json")))
check("greedy temperature 0.0", gen.get("temperature") == 0.0, json.dumps(gen))
check("do_sample false", gen.get("do_sample") is False)
check("eos includes 106 (<end_of_turn>)", 106 in (gen.get("eos_token_id") or []))
check("no top_k/-1 sentinel", "top_k" not in gen and "top_p" not in gen)

tok = AutoTokenizer.from_pretrained(FINAL)
check("tokenizer end_of_turn id 106", tok.convert_tokens_to_ids("<end_of_turn>") == 106)
AutoProcessor.from_pretrained(FINAL)
print("PASS  processor loads")

# weights are bf16 and actually differ from the frozen base snapshot
idx = json.load(open(os.path.join(FINAL, "model.safetensors.index.json")))["weight_map"]
key = "language_model.model.layers.0.mlp.gate_proj.weight"
with safe_open(os.path.join(FINAL, idx[key]), framework="pt") as f:
    w_new = f.get_tensor(key)
base_idx = json.load(open(os.path.join(SNAPSHOT, "model.safetensors.index.json")))["weight_map"]
with safe_open(os.path.join(SNAPSHOT, base_idx[key]), framework="pt") as f:
    w_base = f.get_tensor(key)
check("weights are bf16", w_new.dtype == torch.bfloat16, str(w_new.dtype))
check("weights differ from base snapshot", not torch.equal(w_new, w_base),
      f"mean|delta| = {(w_new.float() - w_base.float()).abs().mean():.3e}")

print("\n" + ("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED"))
sys.exit(0 if ok else 1)
