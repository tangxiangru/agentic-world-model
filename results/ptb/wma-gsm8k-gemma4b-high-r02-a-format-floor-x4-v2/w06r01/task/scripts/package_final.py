#!/usr/bin/env python3
"""Copy a checkpoint into final_model/ and prove it is loadable and complete.

Checks the three things the pitfall catalogue says kill a final model:
weights present, tokenizer present, and a generation_config that decodes greedy
with eos_token_id [1, 106] intact.
"""
import json, os, shutil, sys

src, dst = sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "/home/ben/task/final_model"
if os.path.exists(dst):
    shutil.rmtree(dst)
os.makedirs(dst)
for f in sorted(os.listdir(src)):
    if f in ("training_args.bin", "optimizer.pt", "scheduler.pt", "rng_state.pth",
             "trainer_state.json"):
        continue
    s = os.path.realpath(os.path.join(src, f))
    if os.path.isfile(s):
        shutil.copy(s, os.path.join(dst, f))
print("copied:", sorted(os.listdir(dst)))

gc = json.load(open(os.path.join(dst, "generation_config.json")))
assert gc.get("eos_token_id") == [1, 106], gc
assert gc.get("temperature") == 0.0 and gc.get("do_sample") is False, gc
assert "top_k" not in gc and "top_p" not in gc, gc
cfg = json.load(open(os.path.join(dst, "config.json")))
print("architectures:", cfg["architectures"], "-> evaluate.py model_type:",
      "gemma" if "gemma" in cfg["architectures"][0].lower() else "??")

import torch
from transformers import AutoTokenizer, Gemma3ForConditionalGeneration
tok = AutoTokenizer.from_pretrained(dst)
assert tok.convert_tokens_to_ids("<end_of_turn>") == 106
m = Gemma3ForConditionalGeneration.from_pretrained(dst, dtype=torch.bfloat16,
                                                   device_map="cpu")
n = sum(p.numel() for p in m.parameters())
print(f"loaded on CPU with transformers: {n/1e9:.2f}B params. OK")
