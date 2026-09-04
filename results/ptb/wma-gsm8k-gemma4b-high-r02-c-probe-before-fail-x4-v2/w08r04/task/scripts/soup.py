#!/usr/bin/env python3
"""Uniform weight average of two full fine-tunes of the same frozen base."""
from __future__ import annotations

import argparse
import json
import os
import shutil

import torch
from transformers import AutoProcessor, AutoTokenizer, Gemma3ForConditionalGeneration

BASE = ("/home/ben/hf_cache/hub/models--google--gemma-3-4b-pt/snapshots/"
        "cc012e0a6d0787b4adcc0fa2c4da74402494554d")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--weight-a", type=float, default=0.5)
    ap.add_argument("--dst", required=True)
    args = ap.parse_args()

    wa, wb = args.weight_a, 1.0 - args.weight_a
    ma = Gemma3ForConditionalGeneration.from_pretrained(args.a, dtype=torch.float32)
    sb = Gemma3ForConditionalGeneration.from_pretrained(args.b, dtype=torch.float32).state_dict()
    sa = ma.state_dict()
    assert set(sa) == set(sb), "state dicts differ in keys"
    n = 0
    for k in sa:
        if sa[k].is_floating_point():
            sa[k].mul_(wa).add_(sb[k], alpha=wb)
            n += 1
        else:
            assert torch.equal(sa[k], sb[k]), f"non-float tensor differs: {k}"
    print(f"averaged {n} float tensors, {wa} * A + {wb} * B")
    del sb

    ma = ma.to(torch.bfloat16)
    ma.config.use_cache = True
    ma.save_pretrained(args.dst, safe_serialization=True)
    AutoTokenizer.from_pretrained(BASE).save_pretrained(args.dst)
    try:
        AutoProcessor.from_pretrained(BASE).save_pretrained(args.dst)
    except Exception as e:
        print("processor skipped:", e)
    for f in ("preprocessor_config.json", "processor_config.json"):
        s = os.path.join(BASE, f)
        if os.path.exists(s) and not os.path.exists(os.path.join(args.dst, f)):
            shutil.copy(s, os.path.join(args.dst, f))
    gp = os.path.join(args.dst, "generation_config.json")
    gc = json.load(open(gp))
    gc.update({"temperature": 0.0, "top_p": 1.0, "top_k": 0, "do_sample": False})
    assert 106 in gc["eos_token_id"], gc["eos_token_id"]
    json.dump(gc, open(gp, "w"), indent=2)
    print(json.dumps(gc, indent=2))
    print("wrote", args.dst)


if __name__ == "__main__":
    main()
