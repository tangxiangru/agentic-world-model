#!/usr/bin/env python3
"""Make a Trainer checkpoint loadable by the grader's vLLM.

Casts to bf16, copies the tokenizer and processor files, and writes a
generation_config that stops on <end_of_turn> and decodes greedily.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil

import torch
from transformers import AutoTokenizer, Gemma3ForConditionalGeneration

SNAPSHOT = "/home/ben/hf_cache/hub/models--google--gemma-3-4b-pt/snapshots/cc012e0a6d0787b4adcc0fa2c4da74402494554d"


def finalize(src, dst, temperature=0.0):
    os.makedirs(dst, exist_ok=True)
    m = Gemma3ForConditionalGeneration.from_pretrained(src, dtype=torch.bfloat16)
    m.config.torch_dtype = "bfloat16"
    m.config.use_cache = True
    if hasattr(m.config, "text_config"):
        m.config.text_config.torch_dtype = "bfloat16"
    m.save_pretrained(dst, safe_serialization=True)
    tok = AutoTokenizer.from_pretrained(SNAPSHOT)
    tok.save_pretrained(dst)
    gc = {
        "bos_token_id": 2,
        "eos_token_id": [1, 106],
        "pad_token_id": 0,
        "cache_implementation": "hybrid",
        "do_sample": temperature > 0,
        "temperature": temperature,
    }
    if temperature <= 0:
        gc["temperature"] = 0.0
    json.dump(gc, open(os.path.join(dst, "generation_config.json"), "w"), indent=2)
    for f in ("preprocessor_config.json", "processor_config.json"):
        s = os.path.join(SNAPSHOT, f)
        if os.path.exists(s):
            shutil.copy(s, os.path.join(dst, f))
    print("[finalize] wrote", dst)
    print("[finalize] generation_config:", json.load(open(os.path.join(dst, "generation_config.json"))))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--temperature", type=float, default=0.0)
    a = ap.parse_args()
    finalize(a.src, a.dst, a.temperature)
