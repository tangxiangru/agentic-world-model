#!/usr/bin/env python3
"""Copy a trained checkpoint into final_model/ and verify it loads standalone.

Checks the three things that make a final_model unscorable: missing tokenizer,
a config the grader's model_type() sniffing cannot classify, and a
generation_config that does not stop on <end_of_turn>.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys

os.environ.setdefault("HF_HOME", "/home/ben/hf_cache")

REQUIRED = ["config.json", "generation_config.json", "tokenizer.json",
            "tokenizer_config.json", "special_tokens_map.json"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", default="final_model")
    ap.add_argument("--verify-only", action="store_true")
    args = ap.parse_args()

    if not args.verify_only:
        if os.path.exists(args.dst):
            shutil.rmtree(args.dst)
        shutil.copytree(args.src, args.dst)
        print(f"copied {args.src} -> {args.dst}")

    missing = [f for f in REQUIRED if not os.path.exists(os.path.join(args.dst, f))]
    if missing:
        print("MISSING:", missing)
        sys.exit(1)

    cfg = json.load(open(os.path.join(args.dst, "config.json")))
    arch = cfg["architectures"][0].lower()
    print("architectures:", cfg["architectures"], "-> grader model_type:",
          "gemma" if "gemma" in arch else "UNRECOGNISED")
    gen = json.load(open(os.path.join(args.dst, "generation_config.json")))
    print("generation_config:", gen)
    assert 106 in (gen["eos_token_id"] if isinstance(gen["eos_token_id"], list)
                   else [gen["eos_token_id"]]), "must stop on <end_of_turn> (106)"

    import torch
    from transformers import AutoTokenizer, Gemma3ForConditionalGeneration
    tok = AutoTokenizer.from_pretrained(args.dst)
    print("tokenizer ok, vocab", len(tok))
    m = Gemma3ForConditionalGeneration.from_pretrained(
        args.dst, dtype=torch.bfloat16, device_map="cpu")
    n = sum(p.numel() for p in m.parameters())
    print(f"loaded on CPU: {n/1e9:.2f}B params, dtype {next(m.parameters()).dtype}")
    print("OK")


if __name__ == "__main__":
    main()
