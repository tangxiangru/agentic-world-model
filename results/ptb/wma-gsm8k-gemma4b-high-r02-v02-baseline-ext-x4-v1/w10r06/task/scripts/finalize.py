#!/usr/bin/env python3
"""Copy a trained checkpoint into final_model/ in the shape the grader loads.

The grader does `vllm serve final_model` from a fresh process with
templates/gemma3.jinja as the chat template, and evaluate.py sets no
temperature, so vLLM falls back to the checkpoint's own generation_config.json
for sampling defaults (pitfalls.yaml:final_model_not_loadable).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil

TASK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = "/home/ben/hf_cache/hub/models--google--gemma-3-4b-pt/snapshots/cc012e0a6d0787b4adcc0fa2c4da74402494554d"

# files the trainer does not write but vLLM/transformers want alongside the weights
AUX = ["tokenizer.json", "tokenizer.model", "tokenizer_config.json",
       "special_tokens_map.json", "added_tokens.json",
       "preprocessor_config.json", "processor_config.json"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", default=os.path.join(TASK_DIR, "final_model"))
    ap.add_argument("--greedy", action="store_true",
                    help="write generation_config.json with temperature 0")
    args = ap.parse_args()

    os.makedirs(args.dst, exist_ok=True)
    for f in os.listdir(args.src):
        if f.startswith("optimizer") or f.startswith("rng_state") or f in (
                "scheduler.pt", "trainer_state.json", "training_args.bin"):
            continue
        s = os.path.join(args.src, f)
        if os.path.isfile(s):
            shutil.copy2(s, os.path.join(args.dst, f))
    for f in AUX:
        d = os.path.join(args.dst, f)
        s = os.path.join(BASE, f)
        if not os.path.exists(d) and os.path.exists(s):
            shutil.copy2(s, d)

    gc = json.load(open(os.path.join(BASE, "generation_config.json")))
    if args.greedy:
        gc["do_sample"] = False
        gc["temperature"] = 0.0
        gc.pop("top_k", None)
        gc.pop("top_p", None)
    json.dump(gc, open(os.path.join(args.dst, "generation_config.json"), "w"), indent=2)
    print("generation_config:", json.dumps(gc))
    print("wrote", args.dst)
    print(sorted(os.listdir(args.dst)))


if __name__ == "__main__":
    main()
