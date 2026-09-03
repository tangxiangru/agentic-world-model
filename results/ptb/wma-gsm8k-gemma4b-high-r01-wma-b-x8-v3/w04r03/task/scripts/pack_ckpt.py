#!/usr/bin/env python3
"""Turn a Trainer checkpoint-N/ (fp32 master weights) into a bf16 model dir vLLM can load."""
from __future__ import annotations

import argparse
import json
import os
import shutil

os.environ.setdefault("HF_HOME", "/home/ben/hf_cache")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

BASE = (
    "/home/ben/hf_cache/hub/models--google--gemma-3-4b-pt/snapshots/"
    "cc012e0a6d0787b4adcc0fa2c4da74402494554d"
)

GEN_CFG = {
    "bos_token_id": 2,
    "eos_token_id": [1, 106],
    "pad_token_id": 0,
    "cache_implementation": "hybrid",
    "do_sample": False,
    "temperature": 0.0,
    "transformers_version": "4.50.0.dev0",
}


def pack(src: str, dst: str) -> None:
    m = AutoModelForCausalLM.from_pretrained(src, dtype=torch.bfloat16)
    m.config.use_cache = True
    m.generation_config = GenerationConfig(
        bos_token_id=2, eos_token_id=[1, 106], pad_token_id=0,
        cache_implementation="hybrid", do_sample=False,
    )
    os.makedirs(dst, exist_ok=True)
    m.save_pretrained(dst, safe_serialization=True)
    tok = AutoTokenizer.from_pretrained(BASE)
    tok.chat_template = open("/home/ben/task/templates/gemma3.jinja").read()
    tok.save_pretrained(dst)
    for fn in ["preprocessor_config.json", "processor_config.json", "tokenizer.model"]:
        s = os.path.join(BASE, fn)
        if os.path.exists(s):
            shutil.copy(s, os.path.join(dst, fn))
    json.dump(GEN_CFG, open(os.path.join(dst, "generation_config.json"), "w"), indent=2)
    print("packed", src, "->", dst)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    a = ap.parse_args()
    pack(a.src, a.dst)
