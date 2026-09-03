#!/usr/bin/env python3
"""Assemble final_model/ from a trained checkpoint.

Guards the `final_model_not_loadable` pitfall: the grader loads final_model/ with
vLLM from a fresh process, using templates/gemma3.jinja as the chat template, so
the directory must carry the full Gemma3ForConditionalGeneration side files
(processor/tokenizer), not just the weights the Trainer wrote.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil

BASE = (
    "/home/ben/hf_cache/hub/models--google--gemma-3-4b-pt/snapshots/"
    "cc012e0a6d0787b4adcc0fa2c4da74402494554d"
)
SIDE_FILES = [
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
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", default="/home/ben/task/final_model")
    ap.add_argument(
        "--decode",
        choices=["inherit", "greedy"],
        default="inherit",
        help="'greedy' rewrites generation_config.json so vLLM decodes with "
        "temperature 0 instead of the base checkpoint's top_k 64 / top_p 0.95 sampling",
    )
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    for f in os.listdir(args.ckpt):
        if f in ("optimizer.pt", "scheduler.pt", "rng_state.pth", "trainer_state.json", "training_args.bin"):
            continue
        src = os.path.join(args.ckpt, f)
        if os.path.isfile(src):
            shutil.copy2(src, os.path.join(args.out, f))
    for f in SIDE_FILES:
        dst = os.path.join(args.out, f)
        if not os.path.exists(dst):
            shutil.copy2(os.path.join(BASE, f), dst)

    gc_path = os.path.join(args.out, "generation_config.json")
    gc = json.load(open(gc_path)) if os.path.exists(gc_path) else {}
    if args.decode == "greedy":
        gc.pop("top_k", None)
        gc.pop("top_p", None)
        gc["do_sample"] = False
        gc["temperature"] = 0.0
    # force-repair the eos-collapse pitfall on EVERY staged dir, not just final/
    if gc.get("eos_token_id") != [1, 106]:
        print("REPAIR eos_token_id was", gc.get("eos_token_id"))
        gc["eos_token_id"] = [1, 106]
    gc["bos_token_id"] = 2
    gc["pad_token_id"] = 0
    json.dump(gc, open(gc_path, "w"), indent=2)

    print("wrote", args.out)
    print("generation_config:", json.dumps(gc))
    print(sorted(os.listdir(args.out)))


if __name__ == "__main__":
    main()
