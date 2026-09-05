#!/usr/bin/env python3
"""Write final_model/ from a checkpoint and prove it loads.

Handles both shapes of source directory:
  * a directory already saved by train_sft.py (bf16 weights + tokenizer files)
  * a raw Trainer checkpoint-N directory (fp32 weights, no tokenizer files)

Always writes real files (no symlinks), copies the tokenizer/processor files from
the immutable base snapshot, keeps generation_config.eos_token_id == [1, 106],
and finishes by loading the result from disk on CPU with transformers -- the
final_model_not_loadable pitfall.
"""
import argparse
import json
import os
import shutil

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE = "/home/ben/hf_cache/hub/models--google--gemma-3-4b-pt/snapshots/cc012e0a6d0787b4adcc0fa2c4da74402494554d"
AUX = [
    "added_tokens.json",
    "preprocessor_config.json",
    "processor_config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", default="/home/ben/task/final_model")
    ap.add_argument("--greedy", action="store_true")
    ap.add_argument("--verify-only", action="store_true")
    args = ap.parse_args()

    if not args.verify_only:
        if os.path.exists(args.dst):
            shutil.rmtree(args.dst)
        os.makedirs(args.dst)
        print(f"[load] {args.src}")
        model = AutoModelForCausalLM.from_pretrained(args.src, dtype=torch.bfloat16)
        model.config.use_cache = True
        model.save_pretrained(args.dst, safe_serialization=True)
        for fn in AUX:
            shutil.copy(os.path.join(BASE, fn), os.path.join(args.dst, fn))
        gc_path = os.path.join(args.dst, "generation_config.json")
        gc = json.load(open(gc_path))
        if args.greedy:
            gc["do_sample"] = False
            gc["temperature"] = 0.0
            gc.pop("top_k", None)
            gc.pop("top_p", None)
        assert gc.get("eos_token_id") == [1, 106], gc
        json.dump(gc, open(gc_path, "w"), indent=2)
        del model

    print(f"[verify] loading {args.dst} on CPU from a fresh state")
    tok = AutoTokenizer.from_pretrained(args.dst)
    m = AutoModelForCausalLM.from_pretrained(args.dst, dtype=torch.bfloat16)
    cfg = json.load(open(os.path.join(args.dst, "config.json")))
    gc = json.load(open(os.path.join(args.dst, "generation_config.json")))
    print("  architectures:", cfg["architectures"])
    print("  dtype:", cfg.get("dtype") or cfg.get("torch_dtype"))
    print("  generation_config:", json.dumps(gc))
    print("  params:", sum(p.numel() for p in m.parameters()) / 1e9, "B")
    print("  <end_of_turn> id:", tok.convert_tokens_to_ids("<end_of_turn>"))
    print("  files:", sorted(os.listdir(args.dst)))
    total = sum(
        os.path.getsize(os.path.join(args.dst, f)) for f in os.listdir(args.dst)
    )
    print(f"  size: {total/1e9:.1f} GB")
    print("[verify] OK")


if __name__ == "__main__":
    main()
