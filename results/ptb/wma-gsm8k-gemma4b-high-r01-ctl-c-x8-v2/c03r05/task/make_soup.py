#!/usr/bin/env python3
"""Uniform weight average ("model soup") of two or more checkpoints that share
an initialisation, saved in the shape the grader's vLLM serves."""
from __future__ import annotations

import argparse
import json
import os
import shutil

import torch
from transformers import AutoTokenizer, Gemma3ForConditionalGeneration

BASE = ("/home/ben/hf_cache/hub/models--google--gemma-3-4b-pt/snapshots/"
        "cc012e0a6d0787b4adcc0fa2c4da74402494554d")
END_OF_TURN = "<end_of_turn>"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--weights", nargs="*", type=float, default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    w = args.weights or [1.0 / len(args.models)] * len(args.models)
    assert len(w) == len(args.models)
    w = [x / sum(w) for x in w]
    print("weights", dict(zip(args.models, w)))

    model = Gemma3ForConditionalGeneration.from_pretrained(
        args.models[0], dtype=torch.float32, device_map="cpu")
    sd = model.state_dict()
    for k in sd:
        sd[k] = sd[k] * w[0]
    for path, wi in zip(args.models[1:], w[1:]):
        other = Gemma3ForConditionalGeneration.from_pretrained(
            path, dtype=torch.float32, device_map="cpu").state_dict()
        for k in sd:
            sd[k] += other[k] * wi
        del other
    model.load_state_dict(sd)
    model = model.to(torch.bfloat16)

    os.makedirs(args.out, exist_ok=True)
    model.config.use_cache = True
    # the parents carry the greedy generation_config (do_sample=False +
    # temperature=0.0), which HF refuses to serialise; write a valid one here
    # and the greedy file below.
    from transformers import GenerationConfig
    model.generation_config = GenerationConfig(
        bos_token_id=2, eos_token_id=[1, 106], pad_token_id=0, do_sample=False)
    model.save_pretrained(args.out, safe_serialization=True)
    tok = AutoTokenizer.from_pretrained(BASE)
    tok.chat_template = open("templates/gemma3.jinja").read()
    tok.eos_token = END_OF_TURN
    tok.save_pretrained(args.out)
    shutil.copy2(os.path.join(BASE, "tokenizer.json"),
                 os.path.join(args.out, "tokenizer.json"))
    for fn in ("preprocessor_config.json", "processor_config.json"):
        shutil.copy2(os.path.join(BASE, fn), os.path.join(args.out, fn))
    with open(os.path.join(args.out, "generation_config.json"), "w") as f:
        json.dump({"bos_token_id": 2, "eos_token_id": [1, 106], "pad_token_id": 0,
                   "do_sample": False, "temperature": 0.0,
                   "cache_implementation": "hybrid"}, f, indent=2)
    print("saved", args.out)


if __name__ == "__main__":
    main()
