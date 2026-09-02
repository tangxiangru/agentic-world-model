#!/usr/bin/env python3
"""Make a mid-training checkpoint-<step>/ directory evaluable under the same
decode as the comparator.

Trainer's intermediate saves carry neither tokenizer files nor the greedy
generation_config, so evaluating one as-is would measure a different decode
(T=1.0, top_k 64) than every other number in this session.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil

SNAP = os.environ.get(
    "PTB_BASE_MODEL_SNAPSHOT",
    "/home/ben/hf_cache/hub/models--google--gemma-3-4b-pt/snapshots/"
    "cc012e0a6d0787b4adcc0fa2c4da74402494554d",
)

TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "preprocessor_config.json",
    "processor_config.json",
)

GREEDY = {
    "bos_token_id": 2,
    "cache_implementation": "hybrid",
    # do_sample must stay True for GenerationConfig.save_pretrained to accept
    # temperature/top_k on any later round trip; vLLM reads temperature==0 as greedy.
    "do_sample": True,
    "eos_token_id": [1, 106],
    "pad_token_id": 0,
    "temperature": 0.0,
    "top_k": 0,
    "top_p": 1.0,
    "transformers_version": "4.50.0.dev0",
}


def fix(path: str) -> None:
    for fn in TOKENIZER_FILES:
        src = os.path.join(SNAP, fn)
        dst = os.path.join(path, fn)
        if os.path.exists(src) and not os.path.exists(dst):
            shutil.copyfile(src, dst)
    with open(os.path.join(path, "generation_config.json"), "w") as f:
        json.dump(GREEDY, f, indent=2)
    print(f"fixed {path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    args = ap.parse_args()
    for p in args.paths:
        fix(p)
