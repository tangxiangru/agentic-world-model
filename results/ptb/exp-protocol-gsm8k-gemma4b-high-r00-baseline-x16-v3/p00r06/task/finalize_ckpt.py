#!/usr/bin/env python3
"""Make a Trainer checkpoint-N directory loadable by the grader's vLLM.

Trainer's intermediate checkpoints carry the model weights but not the
tokenizer/processor files, and they inherit the base generation_config
(do_sample=true, top_k=64, top_p=0.95). vLLM reads generation_config.json as
its *default sampling params*, so an untouched checkpoint would be graded
under temperature-1 sampling. This writes a greedy config and copies the
tokenizer/processor files across.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil

FILES = (
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
    "eos_token_id": [1, 106],
    "pad_token_id": 0,
    "cache_implementation": "hybrid",
    "do_sample": False,
    "temperature": 0.0,
    "transformers_version": "4.57.3",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--base", required=True)
    ap.add_argument("--sampling", action="store_true", help="keep the base sampling config instead of greedy")
    args = ap.parse_args()

    for fn in FILES:
        src, dst = os.path.join(args.base, fn), os.path.join(args.ckpt, fn)
        if os.path.exists(src) and not os.path.exists(dst):
            shutil.copy(src, dst)
    gc = os.path.join(args.ckpt, "generation_config.json")
    if args.sampling:
        shutil.copy(os.path.join(args.base, "generation_config.json"), gc)
    else:
        with open(gc, "w") as f:
            json.dump(GREEDY, f, indent=2)
    print("finalized", args.ckpt, "greedy" if not args.sampling else "sampling")


if __name__ == "__main__":
    main()
