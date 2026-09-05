#!/usr/bin/env python3
"""Assemble a servable model directory from a training checkpoint.

Copies the weights + every tokenizer/processor file the grader's vLLM needs and
writes the generation_config.json that decides how the grader decodes (inspect
sends no temperature, so vLLM falls back to this file).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil

SNAPSHOT = (
    "/home/ben/hf_cache/hub/models--google--gemma-3-4b-pt/snapshots/"
    "cc012e0a6d0787b4adcc0fa2c4da74402494554d"
)

NEEDED = [
    "config.json",
    "generation_config.json",
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
    ap.add_argument("--src", required=True, help="training checkpoint dir")
    ap.add_argument("--dst", required=True)
    ap.add_argument("--decode", choices=["base", "greedy"], default="greedy")
    args = ap.parse_args()

    os.makedirs(args.dst, exist_ok=True)
    for fn in os.listdir(args.src):
        if fn.startswith("checkpoint-") or fn in {"optimizer.pt", "scheduler.pt", "rng_state.pth"}:
            continue
        s = os.path.join(args.src, fn)
        if os.path.isfile(s):
            shutil.copy(s, os.path.join(args.dst, fn))
    for fn in NEEDED:
        d = os.path.join(args.dst, fn)
        if not os.path.exists(d) and os.path.exists(os.path.join(SNAPSHOT, fn)):
            shutil.copy(os.path.join(SNAPSHOT, fn), d)

    gc = {"bos_token_id": 2, "eos_token_id": [1, 106], "pad_token_id": 0,
          "cache_implementation": "hybrid"}
    if args.decode == "greedy":
        gc.update({"do_sample": False, "temperature": 0.0, "top_p": 1.0, "top_k": 0})
    else:
        gc.update({"do_sample": True, "top_p": 0.95, "top_k": 64})
    with open(os.path.join(args.dst, "generation_config.json"), "w") as f:
        json.dump(gc, f, indent=2)

    missing = [f for f in NEEDED if not os.path.exists(os.path.join(args.dst, f))]
    print("dst:", args.dst)
    print("files:", sorted(os.listdir(args.dst)))
    print("missing:", missing)


if __name__ == "__main__":
    main()
