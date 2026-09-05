#!/usr/bin/env python3
"""Copy a checkpoint into final_model/, make sure it is complete, and prove it loads.

Guards the final_model_not_loadable pitfall: the grader loads final_model/ with vLLM
from a fresh process, so every tokenizer/processor/config file has to be there and the
generation config has to stop on <end_of_turn>.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys

BASE = "/home/ben/hf_cache/hub/models--google--gemma-3-4b-pt/snapshots/cc012e0a6d0787b4adcc0fa2c4da74402494554d"
NEEDED = [
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "preprocessor_config.json",
    "processor_config.json",
]
SKIP = {"optimizer.pt", "rng_state.pth", "scheduler.pt", "trainer_state.json", "training_args.bin"}

GEN_CFG = {
    "bos_token_id": 2,
    "eos_token_id": [1, 106],
    "pad_token_id": 0,
    "do_sample": False,
    "temperature": 0.0,
    "cache_implementation": "hybrid",
    "transformers_version": "4.57.3",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", default="final_model")
    ap.add_argument("--no-verify", action="store_true")
    args = ap.parse_args()

    if os.path.exists(args.dst):
        shutil.rmtree(args.dst)
    os.makedirs(args.dst)
    for fn in sorted(os.listdir(args.src)):
        if fn in SKIP or os.path.isdir(os.path.join(args.src, fn)):
            continue
        shutil.copy2(os.path.join(args.src, fn), os.path.join(args.dst, fn))
    for fn in NEEDED:
        d = os.path.join(args.dst, fn)
        if not os.path.exists(d) and os.path.exists(os.path.join(BASE, fn)):
            shutil.copy2(os.path.join(BASE, fn), d)
    with open(os.path.join(args.dst, "generation_config.json"), "w") as f:
        json.dump(GEN_CFG, f, indent=2)

    missing = [fn for fn in NEEDED if not os.path.exists(os.path.join(args.dst, fn))]
    if missing:
        print("MISSING:", missing)
        sys.exit(1)
    print("files ok:", sorted(os.listdir(args.dst)))

    if args.no_verify:
        return
    # evaluate.py picks the template from the model path / config architecture
    sys.path.insert(0, os.getcwd())
    cfg = json.load(open(os.path.join(args.dst, "config.json")))
    arch = cfg["architectures"][0].lower()
    assert "gemma" in arch, arch
    print("architecture:", cfg["architectures"][0], "-> template gemma3.jinja")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.dst)
    m = AutoModelForCausalLM.from_pretrained(args.dst, dtype=torch.bfloat16)
    n = sum(p.numel() for p in m.parameters())
    print(f"loaded on CPU with transformers: {n / 1e9:.2f}B params, vocab {len(tok)}")


if __name__ == "__main__":
    main()
