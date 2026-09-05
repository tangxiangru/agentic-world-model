#!/usr/bin/env python3
"""Check final_model/ the way the grader will: fresh process, from disk, no cache tricks.

Covers the `final_model_not_loadable` pitfall: config loadable, weights complete,
tokenizer present, processor files present (vLLM builds an AutoProcessor for
Gemma3ForConditionalGeneration), and the generation_config that vLLM actually
reads (temperature/top_k/top_p) says what we think it says.
"""
import json
import os
import sys

FINAL = sys.argv[1] if len(sys.argv) > 1 else "/home/ben/task/final_model"
ok = True


def check(cond, msg):
    global ok
    print(("PASS  " if cond else "FAIL  ") + msg)
    ok = ok and bool(cond)


files = set(os.listdir(FINAL))
for f in ("config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json",
          "preprocessor_config.json", "processor_config.json", "model.safetensors.index.json"):
    check(f in files, f"{f} present")

cfg = json.load(open(os.path.join(FINAL, "config.json")))
check(cfg["architectures"] == ["Gemma3ForConditionalGeneration"], f"architectures {cfg['architectures']}")

gc = json.load(open(os.path.join(FINAL, "generation_config.json")))
check(gc.get("eos_token_id") == [1, 106], f"eos_token_id {gc.get('eos_token_id')} (106 = <end_of_turn>)")
check(gc.get("temperature") == 0.0, f"temperature {gc.get('temperature')!r} -> vLLM decodes greedily")
check("top_k" not in gc and "top_p" not in gc, "no top_k/top_p left over from the base config")

idx = json.load(open(os.path.join(FINAL, "model.safetensors.index.json")))
shards = set(idx["weight_map"].values())
check(shards <= files, f"all {len(shards)} weight shards present")

from transformers import AutoConfig, AutoTokenizer  # noqa: E402

tok = AutoTokenizer.from_pretrained(FINAL)
check(tok.convert_tokens_to_ids("<end_of_turn>") == 106, "tokenizer maps <end_of_turn> -> 106")
AutoConfig.from_pretrained(FINAL)
check(True, "AutoConfig.from_pretrained works")

import torch  # noqa: E402
from transformers import Gemma3ForConditionalGeneration  # noqa: E402

m = Gemma3ForConditionalGeneration.from_pretrained(FINAL, dtype=torch.bfloat16, device_map="cpu")
n = sum(p.numel() for p in m.parameters())
check(n > 4.0e9, f"loaded on CPU with transformers, {n/1e9:.3f}B params")

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
import fmt  # noqa: E402

tmpl = open("/home/ben/task/templates/gemma3.jinja").read()
mine = fmt.render_prompt("Q?", None)
ref = tok.apply_chat_template([{"role": "user", "content": fmt.user_content("Q?")}],
                              chat_template=tmpl, tokenize=False, add_generation_prompt=True)
check(mine == ref, "training prompt render == grader's gemma3.jinja render")

print("\nALL OK" if ok else "\nPROBLEMS FOUND")
sys.exit(0 if ok else 1)
