#!/usr/bin/env python3
"""Weight-average several checkpoints that share an initialisation."""
from __future__ import annotations

import argparse
import json
import os
import shutil

import torch
from transformers import AutoTokenizer, Gemma3ForConditionalGeneration

SNAPSHOT = "/home/ben/hf_cache/hub/models--google--gemma-3-4b-pt/snapshots/cc012e0a6d0787b4adcc0fa2c4da74402494554d"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--weights", nargs="+", type=float, default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    w = a.weights or [1.0 / len(a.models)] * len(a.models)
    assert len(w) == len(a.models)
    s = sum(w)
    w = [x / s for x in w]
    print("[soup]", list(zip(a.models, w)))

    base = Gemma3ForConditionalGeneration.from_pretrained(a.models[0], dtype=torch.float32)
    acc = {k: v.detach().clone() * w[0] for k, v in base.state_dict().items()}
    for m, wi in zip(a.models[1:], w[1:]):
        other = Gemma3ForConditionalGeneration.from_pretrained(m, dtype=torch.float32)
        sd = other.state_dict()
        for k in acc:
            acc[k] += sd[k] * wi
        del other, sd
    base.load_state_dict(acc)
    base = base.to(torch.bfloat16)
    base.config.torch_dtype = "bfloat16"
    base.config.use_cache = True
    if hasattr(base.config, "text_config"):
        base.config.text_config.torch_dtype = "bfloat16"
    # save_pretrained validates generation_config; the parent's greedy config
    # (do_sample False + temperature 0.0) is rejected by that validator, so drop
    # it here and write the file we actually want afterwards.
    from transformers import GenerationConfig

    base.generation_config = GenerationConfig(bos_token_id=2, eos_token_id=[1, 106], pad_token_id=0)
    os.makedirs(a.out, exist_ok=True)
    base.save_pretrained(a.out, safe_serialization=True)
    AutoTokenizer.from_pretrained(SNAPSHOT).save_pretrained(a.out)
    json.dump(
        {"bos_token_id": 2, "eos_token_id": [1, 106], "pad_token_id": 0,
         "cache_implementation": "hybrid", "do_sample": False, "temperature": 0.0},
        open(os.path.join(a.out, "generation_config.json"), "w"), indent=2,
    )
    for f in ("preprocessor_config.json", "processor_config.json"):
        src = os.path.join(SNAPSHOT, f)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(a.out, f))
    print("[soup] wrote", a.out)


if __name__ == "__main__":
    main()
