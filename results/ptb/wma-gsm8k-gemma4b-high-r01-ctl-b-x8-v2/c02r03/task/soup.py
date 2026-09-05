#!/usr/bin/env python3
"""Uniform weight average ("soup") of two checkpoints that share an ancestor.

exp-03 was initialised from exp-02, so the two sit in the same loss basin and a
parameter-space average is meaningful. The two disagree on 29 of 150 dev items
while scoring the same overall, which is the situation averaging is for.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

BASE = "/home/ben/hf_cache/hub/models--google--gemma-3-4b-pt/snapshots/cc012e0a6d0787b4adcc0fa2c4da74402494554d"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--weights", nargs="+", type=float, default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    w = args.weights or [1.0 / len(args.inputs)] * len(args.inputs)
    assert len(w) == len(args.inputs)
    w = [x / sum(w) for x in w]
    print("mixing", list(zip(args.inputs, w)), flush=True)

    model = AutoModelForCausalLM.from_pretrained(args.inputs[0], dtype=torch.float32)
    sd = model.state_dict()
    for k in sd:
        sd[k].mul_(w[0])

    for path, wi in zip(args.inputs[1:], w[1:]):
        other = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float32)
        osd = other.state_dict()
        assert set(osd) == set(sd), "checkpoints have different parameter sets"
        for k in sd:
            sd[k].add_(osd[k], alpha=wi)
        del other, osd
        print("mixed in", path, flush=True)

    model.load_state_dict(sd)
    model.to(torch.bfloat16)
    model.config.torch_dtype = "bfloat16"
    model.config.dtype = "bfloat16"
    model.generation_config = GenerationConfig(bos_token_id=2, eos_token_id=[1, 106], pad_token_id=0)

    os.makedirs(args.out, exist_ok=True)
    model.save_pretrained(args.out)
    AutoTokenizer.from_pretrained(args.inputs[0]).save_pretrained(args.out)
    gc = json.load(open(os.path.join(BASE, "generation_config.json")))
    gc.update({"do_sample": False, "temperature": 0.0, "top_p": 1.0, "top_k": 0})
    gc.pop("cache_implementation", None)
    json.dump(gc, open(os.path.join(args.out, "generation_config.json"), "w"), indent=2)
    for f in ("preprocessor_config.json", "processor_config.json"):
        src = os.path.join(BASE, f)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(args.out, f))
    print("saved", args.out, flush=True)


if __name__ == "__main__":
    main()
