#!/usr/bin/env python3
"""Uniformly average the weights of two or more checkpoints of the same architecture."""
from __future__ import annotations

import argparse
import json
import os
import shutil

import torch
from safetensors.torch import load_file, save_file
from transformers import AutoTokenizer, Gemma3ForConditionalGeneration


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    models = [Gemma3ForConditionalGeneration.from_pretrained(m, dtype=torch.float32)
              for m in args.models]
    print(f"[soup] averaging {len(models)} checkpoints uniformly")
    sds = [m.state_dict() for m in models]
    base = sds[0]
    keys = set(base)
    for sd in sds[1:]:
        assert set(sd) == keys, "state dicts differ in keys"
    avg = {}
    for k in base:
        acc = base[k].clone()
        for sd in sds[1:]:
            acc += sd[k]
        avg[k] = (acc / len(sds)).to(torch.bfloat16)

    model = models[0]
    model.load_state_dict({k: v.to(torch.float32) for k, v in avg.items()})
    model = model.to(torch.bfloat16)
    # the inputs carry the greedy generation_config vLLM needs, which HF's
    # validator refuses to serialise; neutralise it and rewrite it below
    if getattr(model, "generation_config", None) is not None:
        model.generation_config.temperature = 1.0
        model.generation_config.top_p = None
        model.generation_config.top_k = None
        model.generation_config.do_sample = False
    os.makedirs(args.out, exist_ok=True)
    model.save_pretrained(args.out, safe_serialization=True)
    AutoTokenizer.from_pretrained(args.models[0]).save_pretrained(args.out)
    for f in ("preprocessor_config.json", "processor_config.json"):
        src = os.path.join(args.models[0], f)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(args.out, f))
    json.dump({"bos_token_id": 2, "eos_token_id": [1, 106], "pad_token_id": 0,
               "cache_implementation": "hybrid", "do_sample": False, "temperature": 0.0},
              open(os.path.join(args.out, "generation_config.json"), "w"), indent=2)
    print("[soup] wrote", args.out)


if __name__ == "__main__":
    main()
