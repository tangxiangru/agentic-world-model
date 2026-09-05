#!/usr/bin/env python3
"""Copy a run directory into final_model and set a deterministic generation config."""
from __future__ import annotations

import argparse
import json
import os
import shutil

BASE = os.environ.get(
    "PTB_BASE_MODEL_SNAPSHOT",
    "/home/ben/hf_cache/hub/models--google--gemma-3-4b-pt/snapshots/cc012e0a6d0787b4adcc0fa2c4da74402494554d",
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", default="final_model")
    ap.add_argument("--greedy", action="store_true", default=True)
    ap.add_argument("--sampling", dest="greedy", action="store_false")
    args = ap.parse_args()

    if os.path.exists(args.dst):
        shutil.rmtree(args.dst)
    os.makedirs(args.dst)
    for f in os.listdir(args.src):
        if f.startswith("checkpoint-") or f in ("training_args.bin",):
            continue
        s = os.path.join(args.src, f)
        if os.path.isfile(s):
            shutil.copy(s, os.path.join(args.dst, f))
    # make sure the processor configs are there (vLLM needs them for gemma3 mm)
    for f in ["preprocessor_config.json", "processor_config.json", "tokenizer.model",
              "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
              "added_tokens.json"]:
        d = os.path.join(args.dst, f)
        if not os.path.exists(d) and os.path.exists(os.path.join(BASE, f)):
            shutil.copy(os.path.join(BASE, f), d)

    gc = {
        "bos_token_id": 2,
        "cache_implementation": "hybrid",
        "eos_token_id": [1, 106],
        "pad_token_id": 0,
    }
    if args.greedy:
        gc["do_sample"] = False
        gc["temperature"] = 0.0
    else:
        gc.update({"do_sample": True, "top_k": 64, "top_p": 0.95})
    with open(os.path.join(args.dst, "generation_config.json"), "w") as f:
        json.dump(gc, f, indent=2)
    print("wrote", args.dst)
    print(sorted(os.listdir(args.dst)))


if __name__ == "__main__":
    main()
