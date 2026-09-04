#!/usr/bin/env python3
"""Turn a Trainer checkpoint into a directory vLLM can serve.

Trainer writes fp32 weights and its own generation_config; the grader needs
bf16 weights, the tokenizer/processor files of the parent snapshot, and a
generation_config that makes vLLM decode greedily.

vLLM reads temperature/top_p/top_k out of generation_config.json as its
sampling defaults (evaluate.py sets none of them), and vLLM treats
temperature < 1e-5 as greedy.  transformers refuses to re-save a config with
temperature 0.0 under do_sample False, so 1e-6 under do_sample True is the
value that is both HF-valid and greedy at serving time.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil

import torch
from transformers import AutoConfig, AutoTokenizer

COPY_FILES = [
    "preprocessor_config.json",
    "processor_config.json",
    "added_tokens.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
]
GEN_CFG = {
    "bos_token_id": 2,
    "eos_token_id": [1, 106],
    "pad_token_id": 0,
    "cache_implementation": "hybrid",
    "do_sample": True,
    "temperature": 1e-06,
    "top_p": 1.0,
    "top_k": 0,
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--parent", required=True, help="snapshot with the tokenizer/processor files")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    cfg = AutoConfig.from_pretrained(args.ckpt)
    is_mm = cfg.architectures and "ConditionalGeneration" in cfg.architectures[0]
    if is_mm:
        from transformers import Gemma3ForConditionalGeneration as M
    else:
        from transformers import AutoModelForCausalLM as M
    model = M.from_pretrained(args.ckpt, dtype=torch.bfloat16)
    model.config.use_cache = True
    model.generation_config = type(model.generation_config).from_dict(GEN_CFG)
    model.save_pretrained(args.out, safe_serialization=True)

    tok = AutoTokenizer.from_pretrained(args.parent)
    tok.save_pretrained(args.out)
    for f in COPY_FILES:
        src = os.path.join(args.parent, f)
        dst = os.path.join(args.out, f)
        if os.path.exists(src) and not os.path.exists(dst):
            shutil.copy(src, dst)
    with open(os.path.join(args.out, "generation_config.json"), "w") as f:
        json.dump(GEN_CFG, f, indent=2)
    print("prepared", args.out)
    print(sorted(os.listdir(args.out)))


if __name__ == "__main__":
    main()
