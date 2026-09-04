#!/usr/bin/env python3
"""Make a Trainer checkpoint dir servable by the grader's vLLM, in place.

Trainer's save_model writes weights + config.json only, and it can drop
do_sample/top_k/top_p and collapse eos_token_id [1,106] to 1 in
generation_config.json. 106 is <end_of_turn>, the only terminator we train
(pitfalls.yaml: eos_mismatch). This copies the tokenizer/processor files from the
base snapshot and writes the decode config we mean to serve.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil

FILES = [
    "tokenizer.json", "tokenizer.model", "tokenizer_config.json",
    "special_tokens_map.json", "added_tokens.json",
    "preprocessor_config.json", "processor_config.json",
]

BASE_DECODE = {"do_sample": True, "top_k": 64, "top_p": 0.95}
GREEDY_DECODE = {"do_sample": False, "temperature": 0.0, "top_k": 1, "top_p": 1.0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--decode", choices=["base", "greedy"], default="base")
    args = ap.parse_args()

    for f in FILES:
        src = os.path.join(args.base, f)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(args.ckpt, f))

    gen = {
        "bos_token_id": 2,
        "cache_implementation": "hybrid",
        "eos_token_id": [1, 106],
        "pad_token_id": 0,
    }
    gen.update(BASE_DECODE if args.decode == "base" else GREEDY_DECODE)
    p = os.path.join(args.ckpt, "generation_config.json")
    with open(p, "w") as f:
        json.dump(gen, f, indent=2)
    print("wrote", p, json.dumps(gen))
    print("files:", sorted(os.listdir(args.ckpt)))


if __name__ == "__main__":
    main()
