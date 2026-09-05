#!/usr/bin/env python3
"""Copy a checkpoint into final_model/, force greedy decoding, and verify that a
fresh process can load it the way the grader does."""
from __future__ import annotations

import argparse
import json
import os
import shutil

SNAPSHOT = "/home/ben/hf_cache/hub/models--google--gemma-3-4b-pt/snapshots/cc012e0a6d0787b4adcc0fa2c4da74402494554d"
NEEDED = [
    "config.json",
    "generation_config.json",
    "model.safetensors.index.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "preprocessor_config.json",
    "processor_config.json",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", default="final_model")
    args = ap.parse_args()

    if os.path.exists(args.dst):
        shutil.rmtree(args.dst)
    os.makedirs(args.dst)
    for f in sorted(os.listdir(args.src)):
        if f == "training_args.bin":
            continue
        src = os.path.realpath(os.path.join(args.src, f))
        if os.path.isfile(src):
            shutil.copy2(src, os.path.join(args.dst, f))
    for f in ("preprocessor_config.json", "processor_config.json", "tokenizer.model"):
        if not os.path.exists(os.path.join(args.dst, f)) and os.path.exists(os.path.join(SNAPSHOT, f)):
            shutil.copy2(os.path.join(SNAPSHOT, f), os.path.join(args.dst, f))

    gc_path = os.path.join(args.dst, "generation_config.json")
    g = json.load(open(gc_path))
    g.pop("top_k", None)
    g.pop("top_p", None)
    g["do_sample"] = False
    g["temperature"] = 0.0
    g["eos_token_id"] = [1, 106]
    json.dump(g, open(gc_path, "w"), indent=2)
    print("generation_config:", g)

    missing = [f for f in NEEDED if not os.path.exists(os.path.join(args.dst, f))]
    print("missing files:", missing or "none")

    # the grader's own model_type() path
    cfg = json.load(open(os.path.join(args.dst, "config.json")))
    print("architectures:", cfg["architectures"])

    import torch
    from transformers import AutoTokenizer, Gemma3ForConditionalGeneration

    tok = AutoTokenizer.from_pretrained(args.dst)
    print("tokenizer ok; <end_of_turn> ->", tok.convert_tokens_to_ids("<end_of_turn>"))
    m = Gemma3ForConditionalGeneration.from_pretrained(args.dst, dtype=torch.bfloat16, device_map="cpu")
    n = sum(p.numel() for p in m.parameters())
    print(f"loaded on CPU: {n/1e9:.2f}B params, dtype {next(m.parameters()).dtype}")


if __name__ == "__main__":
    main()
