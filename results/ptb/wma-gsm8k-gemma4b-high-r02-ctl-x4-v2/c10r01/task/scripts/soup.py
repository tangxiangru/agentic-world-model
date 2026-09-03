#!/usr/bin/env python3
"""Uniform (or weighted) parameter average of two or more checkpoints that
share an architecture. Saves a loadable model dir with tokenizer + side-cars."""
from __future__ import annotations

import argparse
import os
import shutil

import torch
from transformers import AutoTokenizer, Gemma3ForConditionalGeneration

SIDECARS = (
    "preprocessor_config.json",
    "processor_config.json",
    "added_tokens.json",
    "special_tokens_map.json",
    "tokenizer.model",
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--weights", nargs="*", type=float, default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    w = args.weights or [1.0 / len(args.models)] * len(args.models)
    assert len(w) == len(args.models)
    s = sum(w)
    w = [x / s for x in w]
    print("weights:", dict(zip(args.models, w)), flush=True)

    base = Gemma3ForConditionalGeneration.from_pretrained(
        args.models[0], dtype=torch.float32, device_map="cpu"
    )
    sd = base.state_dict()
    for k in sd:
        sd[k] = sd[k].float() * w[0]
    for m, wi in zip(args.models[1:], w[1:]):
        other = Gemma3ForConditionalGeneration.from_pretrained(
            m, dtype=torch.float32, device_map="cpu"
        )
        osd = other.state_dict()
        for k in sd:
            sd[k] += osd[k].float() * wi
        del other, osd
    for k in sd:
        sd[k] = sd[k].to(torch.bfloat16)
    base.load_state_dict(sd)
    base.config.dtype = "bfloat16"
    if hasattr(base.config, "text_config"):
        base.config.text_config.dtype = "bfloat16"
    if hasattr(base.config, "vision_config"):
        base.config.vision_config.dtype = "bfloat16"
    base.to(torch.bfloat16)
    # same trap as scripts/train_sft2.py: a greedy generation_config fails
    # GenerationConfig.validate(strict=True) and would abort the save
    from transformers import GenerationConfig

    base.generation_config = GenerationConfig(
        bos_token_id=2, eos_token_id=[1, 106], pad_token_id=0,
        cache_implementation="hybrid",
    )
    try:
        base.save_pretrained(args.out)
    except Exception as e:
        print("[warn] save_pretrained failed:", e, flush=True)
        base.save_pretrained(args.out, safe_serialization=True)
    AutoTokenizer.from_pretrained(args.models[0]).save_pretrained(args.out)
    for fn in SIDECARS:
        src = os.path.join(args.models[0], fn)
        dst = os.path.join(args.out, fn)
        if os.path.exists(src) and not os.path.exists(dst):
            shutil.copy(src, dst)
    import json as _json

    with open(os.path.join(args.out, "generation_config.json"), "w") as gf:
        _json.dump(
            {"bos_token_id": 2, "cache_implementation": "hybrid", "do_sample": False,
             "eos_token_id": [1, 106], "pad_token_id": 0, "temperature": 0.0,
             "top_k": 0, "top_p": 1.0, "transformers_version": "4.57.3"}, gf, indent=2)
    print("saved", args.out, flush=True)


if __name__ == "__main__":
    main()
