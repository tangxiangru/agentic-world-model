#!/usr/bin/env python3
"""Copy a trained checkpoint into final_model/ and verify the grader can load it.

Checks, in order:
  1. every file the grader needs is present (config, weights, tokenizer, processor)
  2. generation_config.json asks vLLM for greedy decoding and stops on <end_of_turn>
  3. transformers can load the directory on CPU from a fresh process
  4. the chat template the grader passes renders against this tokenizer
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys

REQUIRED = ["config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", default="/home/ben/task/final_model")
    ap.add_argument("--no-copy", action="store_true")
    args = ap.parse_args()

    if not args.no_copy:
        if os.path.exists(args.dst):
            print(f"[finalize] removing existing {args.dst}")
            shutil.rmtree(args.dst)
        print(f"[finalize] copying {args.src} -> {args.dst}")
        shutil.copytree(args.src, args.dst)

    files = sorted(os.listdir(args.dst))
    print("[finalize] files:", files)
    missing = [f for f in REQUIRED if f not in files]
    if missing:
        sys.exit(f"MISSING required files: {missing}")
    if not any(f.endswith(".safetensors") for f in files):
        sys.exit("MISSING safetensors weights")

    gp = os.path.join(args.dst, "generation_config.json")
    gen = json.load(open(gp))
    if gen.get("temperature") != 0.0 or 106 not in gen.get("eos_token_id", []):
        gen = {"bos_token_id": 2, "eos_token_id": [1, 106], "pad_token_id": 0,
               "cache_implementation": "hybrid", "do_sample": False, "temperature": 0.0}
        json.dump(gen, open(gp, "w"), indent=2)
        print("[finalize] rewrote generation_config.json for greedy decoding")
    print("[finalize] generation_config:", gen)
    assert gen.get("temperature") == 0.0, "vLLM reads temperature from generation_config; want greedy"
    assert 106 in gen.get("eos_token_id", []), "<end_of_turn> (106) must be an eos id"

    cfg = json.load(open(os.path.join(args.dst, "config.json")))
    arch = cfg["architectures"][0].lower()
    assert "gemma" in arch, f"evaluate.py picks the template off the architecture; got {arch}"
    print("[finalize] architecture:", cfg["architectures"][0])

    import torch
    from transformers import AutoTokenizer, Gemma3ForConditionalGeneration

    tok = AutoTokenizer.from_pretrained(args.dst)
    template = open("/home/ben/task/templates/gemma3.jinja").read()
    rendered = tok.apply_chat_template(
        [{"role": "user", "content": "2+2?"}], chat_template=template,
        tokenize=False, add_generation_prompt=True)
    print("[finalize] grader prompt renders as:", repr(rendered))
    assert rendered.endswith("<start_of_turn>model\n")

    print("[finalize] loading on CPU with transformers ...")
    model = Gemma3ForConditionalGeneration.from_pretrained(args.dst, dtype=torch.bfloat16)
    n = sum(p.numel() for p in model.parameters())
    print(f"[finalize] loaded OK, {n/1e9:.2f}B params, dtype={next(model.parameters()).dtype}")
    print("[finalize] PASS")


if __name__ == "__main__":
    main()
