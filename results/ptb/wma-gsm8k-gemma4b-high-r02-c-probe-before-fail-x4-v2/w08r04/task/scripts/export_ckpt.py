#!/usr/bin/env python3
"""Turn a mid-training Trainer checkpoint (fp32 weights, no tokenizer) into a
directory the grader's vLLM can load: bf16 weights + tokenizer + generation_config.
Optionally pins greedy decode.
"""
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
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--greedy", action="store_true")
    args = ap.parse_args()

    m = Gemma3ForConditionalGeneration.from_pretrained(args.src, dtype=torch.bfloat16)
    m.config.use_cache = True
    m.save_pretrained(args.dst, safe_serialization=True)
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
    gc = json.load(open(gp)) if os.path.exists(gp) else json.load(
        open(os.path.join(BASE, "generation_config.json")))
    if args.greedy:
        gc.update({"temperature": 0.0, "top_p": 1.0, "top_k": 0, "do_sample": False})
    eos = gc.get("eos_token_id")
    assert isinstance(eos, list) and 106 in eos, f"eos lost <end_of_turn>: {eos!r}"
    json.dump(gc, open(gp, "w"), indent=2)
    print(json.dumps(gc, indent=2))
    print("wrote", args.dst)


if __name__ == "__main__":
    main()
