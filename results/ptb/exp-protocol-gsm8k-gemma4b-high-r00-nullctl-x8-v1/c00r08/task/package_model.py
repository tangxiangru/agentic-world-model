"""Turn a training checkpoint into an eval-ready model directory.

Copies weights + config, restores the pristine gemma-3 tokenizer/processor files,
and writes a greedy generation_config.json (inspect sends no sampling params, so
vLLM falls back to whatever the model dir declares).
"""
import argparse
import json
import os
import shutil

from common import BASE_SNAPSHOT

WEIGHT_FILES = ("config.json",)
FROM_BASE = (
    "preprocessor_config.json",
    "processor_config.json",
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
)

ap = argparse.ArgumentParser()
ap.add_argument("src")
ap.add_argument("dst")
ap.add_argument("--temperature", type=float, default=0.0)
a = ap.parse_args()

if os.path.exists(a.dst):
    shutil.rmtree(a.dst)
os.makedirs(a.dst)

# GRPO checkpoints come out in fp32; re-save in bf16 so the model still fits
# under evaluate.py's default --gpu-memory-utilization 0.3
import torch
from transformers import Gemma3ForConditionalGeneration

model = Gemma3ForConditionalGeneration.from_pretrained(a.src, dtype=torch.bfloat16)
model.config.torch_dtype = "bfloat16"
model.save_pretrained(a.dst)
del model
for fn in FROM_BASE:
    src = os.path.join(BASE_SNAPSHOT, fn)
    if os.path.exists(src):
        shutil.copy(src, os.path.join(a.dst, fn))

gen = {
    "bos_token_id": 2,
    "cache_implementation": "hybrid",
    "eos_token_id": [1, 106],
    "pad_token_id": 0,
    "do_sample": a.temperature > 0,
    "temperature": a.temperature,
    "top_p": 1.0,
    "top_k": -1,
    "transformers_version": "4.50.0.dev0",
}
with open(os.path.join(a.dst, "generation_config.json"), "w") as f:
    json.dump(gen, f, indent=2)

print("packaged", a.src, "->", a.dst)
print(sorted(os.listdir(a.dst)))
