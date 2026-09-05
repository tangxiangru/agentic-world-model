#!/usr/bin/env python3
"""Make a Trainer intermediate checkpoint loadable by the grader.

Trainer's checkpoint-<n>/ has the weights (in the training dtype) and config,
but no processor configs and no greedy generation_config, so it would be read
under different decode than ckpts/*/final. This writes both, and optionally
casts the weights to bf16, so an intermediate checkpoint is scored under
exactly the protocol final/ is scored under.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

GEN = {
    "bos_token_id": 2,
    "eos_token_id": [1, 106],
    "pad_token_id": 0,
    "do_sample": False,
    "temperature": 0.0,
    "top_p": 1.0,
    "top_k": 0,
    "transformers_version": "4.57.3",
}
COPY = (
    "preprocessor_config.json",
    "processor_config.json",
    "added_tokens.json",
    "special_tokens_map.json",
    "tokenizer.model",
    "tokenizer.json",
    "tokenizer_config.json",
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--out", required=True)
    ap.add_argument("--base", default=os.environ["PTB_BASE_MODEL_SNAPSHOT"])
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    model = AutoModelForCausalLM.from_pretrained(args.ckpt, dtype=torch.bfloat16)
    model.config.use_cache = True
    model.save_pretrained(args.out, safe_serialization=True)
    AutoTokenizer.from_pretrained(args.base).save_pretrained(args.out)
    for fn in COPY:
        src = os.path.join(args.base, fn)
        dst = os.path.join(args.out, fn)
        if os.path.exists(src) and not os.path.exists(dst):
            shutil.copy(src, dst)
    with open(os.path.join(args.out, "generation_config.json"), "w") as f:
        json.dump(GEN, f, indent=2)
    print("wrote", args.out, sorted(os.listdir(args.out)))


if __name__ == "__main__":
    main()
