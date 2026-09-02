#!/usr/bin/env python3
"""Assemble final_model/ from a trained checkpoint and prove it loads.

Guards the final_model_not_loadable pitfall:
  * copies the processor/preprocessor configs the base snapshot ships - the grader
    serves this as Gemma3ForConditionalGeneration, and vLLM resolves an AutoProcessor
    for multimodal architectures, which needs preprocessor_config.json
  * copies the tokenizer if the training save left it out
  * optionally rewrites generation_config.json (vLLM's --generation-config defaults
    to 'auto', so this file *is* the served sampling policy)
  * loads the result from disk on CPU with transformers before declaring success
"""
from __future__ import annotations

import argparse
import json
import os
import shutil

SNAP = os.environ["PTB_BASE_MODEL_SNAPSHOT"]
NEEDED_FROM_BASE = [
    "preprocessor_config.json",
    "processor_config.json",
    "added_tokens.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="trained checkpoint dir")
    ap.add_argument("--dst", default="final_model")
    ap.add_argument(
        "--greedy",
        action="store_true",
        help="write temperature 0 into generation_config.json (vLLM then decodes greedily)",
    )
    ap.add_argument("--no-verify", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.dst, exist_ok=True)
    for f in os.listdir(args.src):
        if f in ("train_summary.json",) or f.startswith("optimizer"):
            continue
        s = os.path.join(args.src, f)
        if os.path.isfile(s):
            shutil.copy2(s, os.path.join(args.dst, f))
    for f in NEEDED_FROM_BASE:
        d = os.path.join(args.dst, f)
        s = os.path.join(SNAP, f)
        if not os.path.exists(d) and os.path.exists(s):
            shutil.copy2(s, d)
            print("copied from base snapshot:", f)

    gc_path = os.path.join(args.dst, "generation_config.json")
    gc = json.load(open(gc_path)) if os.path.exists(gc_path) else {}
    gc.setdefault("bos_token_id", 2)
    gc.setdefault("pad_token_id", 0)
    gc["eos_token_id"] = [1, 106]
    if args.greedy:
        gc["do_sample"] = False
        gc["temperature"] = 0.0
        gc.pop("top_k", None)
        gc.pop("top_p", None)
    json.dump(gc, open(gc_path, "w"), indent=2)
    print("generation_config:", gc)

    cfg = json.load(open(os.path.join(args.dst, "config.json")))
    print("architectures:", cfg["architectures"], "dtype:", cfg.get("dtype", cfg.get("torch_dtype")))
    print("files:", sorted(os.listdir(args.dst)))

    if not args.no_verify:
        import torch
        from transformers import AutoProcessor, AutoTokenizer, Gemma3ForConditionalGeneration

        tok = AutoTokenizer.from_pretrained(args.dst)
        print("tokenizer ok, eos", tok.eos_token, "vocab", len(tok))
        try:
            AutoProcessor.from_pretrained(args.dst)
            print("AutoProcessor ok")
        except Exception as e:  # noqa: BLE001
            print("AutoProcessor FAILED:", e)
        m = Gemma3ForConditionalGeneration.from_pretrained(
            args.dst, dtype=torch.bfloat16, device_map="cpu"
        )
        n = sum(p.numel() for p in m.parameters())
        print(f"model loads on CPU: {n/1e9:.2f}B params, dtype {next(m.parameters()).dtype}")


if __name__ == "__main__":
    main()
