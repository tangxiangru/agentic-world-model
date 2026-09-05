#!/usr/bin/env python3
"""Materialize a trained checkpoint as a deployable model dir (bf16 + greedy generation config)."""
from __future__ import annotations

import argparse
import json
import os
import shutil

import torch
from transformers import AutoProcessor, AutoTokenizer, Gemma3ForConditionalGeneration

SNAPSHOT = os.environ["PTB_BASE_MODEL_SNAPSHOT"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--temperature", type=float, default=0.0)
    args = ap.parse_args()

    if os.path.exists(args.dst):
        shutil.rmtree(args.dst)
    os.makedirs(args.dst, exist_ok=True)

    model = Gemma3ForConditionalGeneration.from_pretrained(args.src, dtype=torch.bfloat16)
    model.config.torch_dtype = "bfloat16"
    from transformers import GenerationConfig
    model.generation_config = GenerationConfig(
        bos_token_id=2, eos_token_id=[1, 106], pad_token_id=0,
        cache_implementation="hybrid", do_sample=True,
        temperature=1e-6, top_p=1.0, top_k=1)
    model.save_pretrained(args.dst, safe_serialization=True)

    tok = AutoTokenizer.from_pretrained(SNAPSHOT)
    tok.save_pretrained(args.dst)
    # processor files keep the multimodal config self-consistent for vLLM
    try:
        proc = AutoProcessor.from_pretrained(SNAPSHOT)
        proc.save_pretrained(args.dst)
    except Exception as e:  # pragma: no cover
        print("processor copy skipped:", e)

    gen = {
        "bos_token_id": 2,
        "eos_token_id": [1, 106],
        "pad_token_id": 0,
        "cache_implementation": "hybrid",
        "do_sample": True,
        "temperature": max(args.temperature, 1e-6),
        "top_p": 1.0,
        "top_k": 1,
        "transformers_version": "4.57.3",
    }
    with open(os.path.join(args.dst, "generation_config.json"), "w") as f:
        json.dump(gen, f, indent=2)
    print("wrote", args.dst)


if __name__ == "__main__":
    main()
