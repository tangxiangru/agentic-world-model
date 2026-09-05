#!/usr/bin/env python3
"""Make a Trainer checkpoint directory servable by the grader's vLLM.

Copies the tokenizer/processor files from the base snapshot (Trainer's
intermediate checkpoints only contain weights + config) and optionally writes a
greedy generation_config.json.  vLLM 0.11 runs with `--generation-config auto`,
so whatever is in generation_config.json becomes the server's default sampling
params.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil

BASE = os.environ.get(
    "PTB_BASE_MODEL_SNAPSHOT",
    "/home/ben/hf_cache/hub/models--google--gemma-3-4b-pt/snapshots/cc012e0a6d0787b4adcc0fa2c4da74402494554d",
)
AUX = [
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "preprocessor_config.json",
    "processor_config.json",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--greedy", type=int, default=1)
    ap.add_argument("--base", default=BASE)
    args = ap.parse_args()

    for f in AUX:
        src = os.path.join(args.base, f)
        dst = os.path.join(args.ckpt, f)
        if os.path.exists(src) and not os.path.exists(dst):
            shutil.copy2(src, dst)
            print("copied", f)

    gc_path = os.path.join(args.ckpt, "generation_config.json")
    gc = json.load(open(os.path.join(args.base, "generation_config.json")))
    if args.greedy:
        gc.update({"do_sample": False, "temperature": 0.0})
        gc.pop("top_k", None)
        gc.pop("top_p", None)
    json.dump(gc, open(gc_path, "w"), indent=2)
    print("wrote", gc_path, gc)


if __name__ == "__main__":
    main()
