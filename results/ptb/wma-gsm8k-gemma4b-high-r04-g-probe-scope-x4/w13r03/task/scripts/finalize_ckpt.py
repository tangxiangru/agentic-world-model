#!/usr/bin/env python3
"""Turn a Trainer output directory into something the grader's vLLM can load.

  * casts the weights to bf16 (Trainer saves in the fp32 training dtype)
  * copies the tokenizer / processor files from the base snapshot
  * writes the session's frozen greedy generation_config
    (evaluate.py sends no temperature, so vLLM reads it from here)
"""
from __future__ import annotations

import argparse
import json
import os
import shutil

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE = "/home/ben/hf_cache/hub/models--google--gemma-3-4b-pt/snapshots/cc012e0a6d0787b4adcc0fa2c4da74402494554d"

GREEDY = {
    "bos_token_id": 2,
    "cache_implementation": "hybrid",
    "do_sample": False,
    "temperature": 0.0,
    "eos_token_id": [1, 106],
    "pad_token_id": 0,
    "transformers_version": "4.50.0.dev0",
}


def _force_bf16_config(path: str) -> None:
    """save_pretrained leaves text_config.dtype at the fp32 training dtype even
    after model.to(bfloat16); vLLM would then load the bf16 shards upcast."""
    c = json.load(open(path))
    c["dtype"] = c["torch_dtype"] = "bfloat16"
    for k in ("text_config", "vision_config"):
        if k in c:
            c[k]["dtype"] = c[k]["torch_dtype"] = "bfloat16"
    json.dump(c, open(path, "w"), indent=2)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    args = ap.parse_args()

    model = AutoModelForCausalLM.from_pretrained(args.src, dtype=torch.bfloat16)
    model.config.use_cache = True
    os.makedirs(args.dst, exist_ok=True)
    model.save_pretrained(args.dst, safe_serialization=True)
    _force_bf16_config(os.path.join(args.dst, "config.json"))
    AutoTokenizer.from_pretrained(BASE).save_pretrained(args.dst)
    for f in ("preprocessor_config.json", "processor_config.json"):
        src = os.path.join(BASE, f)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(args.dst, f))
    with open(os.path.join(args.dst, "generation_config.json"), "w") as f:
        json.dump(GREEDY, f, indent=2)
    print(f"[finalize] {args.src} -> {args.dst}")
    print(sorted(os.listdir(args.dst)))


if __name__ == "__main__":
    main()
