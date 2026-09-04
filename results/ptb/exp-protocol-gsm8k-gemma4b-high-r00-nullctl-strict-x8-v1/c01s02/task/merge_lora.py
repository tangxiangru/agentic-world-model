#!/usr/bin/env python3
"""Merge a LoRA adapter into the base checkpoint and write a deployable model dir."""
from __future__ import annotations

import argparse
import json
import os
import shutil

import torch
from peft import PeftModel
from transformers import AutoProcessor, AutoTokenizer, Gemma3ForConditionalGeneration

SNAPSHOT = os.environ["PTB_BASE_MODEL_SNAPSHOT"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--dst", required=True)
    args = ap.parse_args()

    if os.path.exists(args.dst):
        shutil.rmtree(args.dst)
    os.makedirs(args.dst)

    model = Gemma3ForConditionalGeneration.from_pretrained(args.base, dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(model, args.adapter)
    model = model.merge_and_unload()
    model.config.torch_dtype = "bfloat16"
    from transformers import GenerationConfig
    model.generation_config = GenerationConfig(
        bos_token_id=2, eos_token_id=[1, 106], pad_token_id=0,
        cache_implementation="hybrid", do_sample=True,
        temperature=1e-6, top_p=1.0, top_k=1)
    model.save_pretrained(args.dst, safe_serialization=True)

    tok = AutoTokenizer.from_pretrained(SNAPSHOT)
    tok.save_pretrained(args.dst)
    try:
        AutoProcessor.from_pretrained(SNAPSHOT).save_pretrained(args.dst)
    except Exception as e:
        print("processor copy skipped:", e)

    gen = {
        "bos_token_id": 2,
        "eos_token_id": [1, 106],
        "pad_token_id": 0,
        "cache_implementation": "hybrid",
        "do_sample": True,
        "temperature": 1e-06,
        "top_p": 1.0,
        "top_k": 1,
        "transformers_version": "4.57.3",
    }
    with open(os.path.join(args.dst, "generation_config.json"), "w") as f:
        json.dump(gen, f, indent=2)
    print("wrote", args.dst)


if __name__ == "__main__":
    main()
