#!/usr/bin/env python3
"""Materialise an evaluation-ready copy of a checkpoint.

Copies weights + tokenizer into `--dst` and writes the decode configuration
into generation_config.json, which is what vLLM reads for its default
sampling params (vllm ModelConfig.generation_config="auto").
"""
from __future__ import annotations

import argparse
import json
import os
import shutil

KEEP = {
    "config.json",
    "generation_config.json",
    "model.safetensors.index.json",
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "preprocessor_config.json",
    "processor_config.json",
    "chat_template.jinja",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--decode", choices=["greedy", "sampled"], default="greedy")
    ap.add_argument("--link", action="store_true", help="hardlink the safetensors instead of copying")
    args = ap.parse_args()

    os.makedirs(args.dst, exist_ok=True)
    for fn in sorted(os.listdir(args.src)):
        src = os.path.join(args.src, fn)
        if not os.path.isfile(src):
            continue
        if not (fn in KEEP or fn.endswith(".safetensors")):
            continue
        dst = os.path.join(args.dst, fn)
        if os.path.exists(dst):
            os.remove(dst)
        if args.link and fn.endswith(".safetensors"):
            os.link(src, dst)
        else:
            shutil.copy2(src, dst)

    gc_path = os.path.join(args.dst, "generation_config.json")
    gc = json.load(open(gc_path)) if os.path.exists(gc_path) else {}
    gc.update({"bos_token_id": 2, "eos_token_id": [1, 106], "pad_token_id": 0,
               "cache_implementation": "hybrid"})
    if args.decode == "greedy":
        gc.update({"do_sample": False, "temperature": 0.0})
        gc.pop("top_k", None)
        gc.pop("top_p", None)
    else:
        gc.update({"do_sample": True, "top_k": 64, "top_p": 0.95})
        gc.pop("temperature", None)
    json.dump(gc, open(gc_path, "w"), indent=2)
    print("wrote", gc_path, json.dumps(gc))
    print("files:", sorted(os.listdir(args.dst)))


if __name__ == "__main__":
    main()
