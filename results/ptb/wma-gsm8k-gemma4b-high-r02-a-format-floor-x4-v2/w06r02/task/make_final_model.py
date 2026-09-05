#!/usr/bin/env python3
"""Assemble final_model/ from a checkpoint: real files (no symlinks), tokenizer and
processor configs present, generation_config pinned to greedy with eos [1, 106].
Then load it on CPU with transformers to prove the grader's fresh process can load it."""
import argparse
import json
import os
import shutil

ap = argparse.ArgumentParser()
ap.add_argument("--src", required=True)
ap.add_argument("--dst", default="final_model")
ap.add_argument("--temperature", type=float, default=0.0)
ap.add_argument("--no-verify", action="store_true")
a = ap.parse_args()

os.makedirs(a.dst, exist_ok=True)
for f in sorted(os.listdir(a.src)):
    src = os.path.realpath(os.path.join(a.src, f))
    if not os.path.isfile(src) or f == "training_args.bin":
        continue
    shutil.copy(src, os.path.join(a.dst, f))
    print("copied", f)

gc = json.load(open(os.path.join(a.dst, "generation_config.json")))
gc.pop("top_k", None)
gc.pop("top_p", None)
gc["do_sample"] = False
gc["temperature"] = a.temperature
gc["eos_token_id"] = [1, 106]
gc["bos_token_id"] = 2
gc["pad_token_id"] = 0
json.dump(gc, open(os.path.join(a.dst, "generation_config.json"), "w"), indent=2)
print("generation_config:", json.dumps(gc))

need = ["config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json",
        "special_tokens_map.json", "preprocessor_config.json", "processor_config.json",
        "model.safetensors.index.json"]
missing = [f for f in need if not os.path.exists(os.path.join(a.dst, f))]
print("missing:", missing)
assert not missing, missing

if not a.no_verify:
    import torch
    from transformers import AutoTokenizer, Gemma3ForConditionalGeneration

    tok = AutoTokenizer.from_pretrained(a.dst)
    m = Gemma3ForConditionalGeneration.from_pretrained(a.dst, torch_dtype=torch.bfloat16)
    n = sum(p.numel() for p in m.parameters())
    print("loaded on CPU:", type(m).__name__, f"{n/1e9:.2f}B params, dtype", m.dtype)
    tpl = open("templates/gemma3.jinja").read()
    s = tok.apply_chat_template([{"role": "user", "content": "hi"}], chat_template=tpl,
                                tokenize=False, add_generation_prompt=True)
    print("template renders:", repr(s))
print("final_model ready at", a.dst)
