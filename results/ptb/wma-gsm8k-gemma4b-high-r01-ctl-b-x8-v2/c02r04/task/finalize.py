#!/usr/bin/env python3
"""Copy the chosen checkpoint into final_model/ and prove it is loadable.

Guards the final_model_not_loadable pitfall: real files (not symlinks), the
tokenizer and processor files beside the weights, a greedy generation_config,
and a fresh-process transformers load on CPU before the deadline.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys

NEEDED = [
    "config.json", "generation_config.json", "model.safetensors.index.json",
    "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
    "preprocessor_config.json", "processor_config.json",
]

ap = argparse.ArgumentParser()
ap.add_argument("--src", required=True)
ap.add_argument("--dst", default="final_model")
args = ap.parse_args()

if os.path.exists(args.dst):
    shutil.rmtree(args.dst)
os.makedirs(args.dst)
for fn in sorted(os.listdir(args.src)):
    src = os.path.realpath(os.path.join(args.src, fn))
    if os.path.isfile(src) and fn != "training_args.bin":
        shutil.copy(src, os.path.join(args.dst, fn))

# greedy decoding: evaluate.py sends no temperature, so vLLM's defaults come
# from this file (exp-03)
with open(os.path.join(args.dst, "generation_config.json"), "w") as f:
    json.dump({
        "bos_token_id": 2,
        "eos_token_id": [1, 106],
        "pad_token_id": 0,
        "cache_implementation": "hybrid",
        "do_sample": False,
        "temperature": 0.0,
        "transformers_version": "4.57.3",
    }, f, indent=2)

missing = [f for f in NEEDED if not os.path.exists(os.path.join(args.dst, f))]
links = [f for f in os.listdir(args.dst)
         if os.path.islink(os.path.join(args.dst, f))]
print("files:", len(os.listdir(args.dst)), "missing:", missing, "symlinks:", links)

# fresh process, CPU only, meta-device init then one real forward on 2 tokens
probe = f'''
import torch, json
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
p = "{os.path.abspath(args.dst)}"
cfg = AutoConfig.from_pretrained(p)
print("arch", cfg.architectures)
tok = AutoTokenizer.from_pretrained(p)
print("eos", tok.eos_token, "end_of_turn id", tok.convert_tokens_to_ids("<end_of_turn>"))
m = AutoModelForCausalLM.from_pretrained(p, dtype=torch.float32, device_map="cpu")
ids = tok("<bos><start_of_turn>user\\nhi<end_of_turn>\\n<start_of_turn>model\\n",
          add_special_tokens=False, return_tensors="pt")
with torch.no_grad():
    out = m(**ids)
print("logits", tuple(out.logits.shape))
print("LOAD_OK")
'''
r = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True,
                   env={**os.environ, "CUDA_VISIBLE_DEVICES": ""})
print(r.stdout[-1500:])
if r.returncode != 0 or "LOAD_OK" not in r.stdout:
    print(r.stderr[-3000:])
    sys.exit(1)
print("final_model is loadable from a fresh process")
