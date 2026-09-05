#!/usr/bin/env python3
"""Install a trained checkpoint as final_model/ and verify it loads standalone."""
from __future__ import annotations

import argparse
import json
import os
import shutil

REQUIRED = ["config.json", "generation_config.json", "model.safetensors.index.json",
            "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
            "preprocessor_config.json", "processor_config.json"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", default="/home/ben/task/final_model")
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args()

    if os.path.exists(args.dst):
        shutil.rmtree(args.dst)
    shutil.copytree(args.src, args.dst)

    missing = [f for f in REQUIRED if not os.path.exists(os.path.join(args.dst, f))]
    print("missing:", missing)
    assert not missing, missing
    cfg = json.load(open(os.path.join(args.dst, "config.json")))
    gen = json.load(open(os.path.join(args.dst, "generation_config.json")))
    print("architectures:", cfg["architectures"])
    print("generation_config:", gen)
    assert "gemma" in cfg["architectures"][0].lower()
    assert gen.get("temperature") == 0.0 and gen.get("do_sample") is False
    assert 106 in gen["eos_token_id"]

    if args.verify:
        import torch
        from transformers import AutoTokenizer, Gemma3ForConditionalGeneration
        tok = AutoTokenizer.from_pretrained(args.dst)
        m = Gemma3ForConditionalGeneration.from_pretrained(
            args.dst, dtype=torch.bfloat16, device_map="cpu")
        n = sum(p.numel() for p in m.parameters())
        print(f"loaded on CPU with transformers: {n/1e9:.3f}B params, "
              f"vocab {len(tok)}")
    print("OK", args.dst)


if __name__ == "__main__":
    main()
