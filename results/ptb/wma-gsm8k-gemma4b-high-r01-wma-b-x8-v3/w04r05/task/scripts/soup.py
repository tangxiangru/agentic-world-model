#!/usr/bin/env python3
"""Uniform weight average of two full-parameter checkpoints of the same model.

Both parents descend from the same base and the same trajectory (exp-04 is a
continuation of exp-02), so the weights are in the same basin and a plain
average is well defined.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil

import torch
from transformers import AutoTokenizer, Gemma3ForConditionalGeneration

BASE_SNAPSHOT = (
    "/home/ben/hf_cache/hub/models--google--gemma-3-4b-pt/snapshots/"
    "cc012e0a6d0787b4adcc0fa2c4da74402494554d"
)
AUX = ["tokenizer.json", "tokenizer.model", "tokenizer_config.json",
       "special_tokens_map.json", "added_tokens.json",
       "preprocessor_config.json", "processor_config.json"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--weight-a", type=float, default=0.5)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    wa, wb = args.weight_a, 1.0 - args.weight_a
    ma = Gemma3ForConditionalGeneration.from_pretrained(args.a, dtype=torch.float32)
    mb = Gemma3ForConditionalGeneration.from_pretrained(args.b, dtype=torch.float32)
    sa, sb = ma.state_dict(), mb.state_dict()
    assert set(sa) == set(sb), "checkpoints have different parameter sets"
    with torch.no_grad():
        for k in sa:
            sa[k].mul_(wa).add_(sb[k], alpha=wb)
    ma.load_state_dict(sa)
    ma = ma.to(torch.bfloat16)
    # A parent written by train_sft.save_full carries do_sample:false together
    # with temperature:0.0, and GenerationConfig.save_pretrained refuses that
    # pair -- it killed the first run of this script. Reset before saving; the
    # greedy JSON is written over the top afterwards.
    from transformers import GenerationConfig

    ma.generation_config = GenerationConfig(
        bos_token_id=2, eos_token_id=[1, 106], pad_token_id=0,
        cache_implementation="hybrid",
    )

    os.makedirs(args.out, exist_ok=True)
    ma.save_pretrained(args.out, safe_serialization=True)
    AutoTokenizer.from_pretrained(BASE_SNAPSHOT).save_pretrained(args.out)
    for fn in AUX:
        src, dst = os.path.join(BASE_SNAPSHOT, fn), os.path.join(args.out, fn)
        if os.path.exists(src) and not os.path.exists(dst):
            shutil.copyfile(src, dst)
    json.dump(
        {"bos_token_id": 2, "eos_token_id": [1, 106], "pad_token_id": 0,
         "cache_implementation": "hybrid", "do_sample": False, "temperature": 0.0},
        open(os.path.join(args.out, "generation_config.json"), "w"), indent=2,
    )
    print(f"soup {wa}*{args.a} + {wb}*{args.b} -> {args.out}")


if __name__ == "__main__":
    main()
