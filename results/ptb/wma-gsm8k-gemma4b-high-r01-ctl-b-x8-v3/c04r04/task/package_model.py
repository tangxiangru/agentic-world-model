#!/usr/bin/env python3
"""Package a Trainer checkpoint into a directory vLLM can serve:
bf16 weights + tokenizer + processor files + an explicit generation_config.
"""
import argparse, json, os, shutil

import torch
from transformers import AutoModelForCausalLM

BASE = "/home/ben/hf_cache/hub/models--google--gemma-3-4b-pt/snapshots/cc012e0a6d0787b4adcc0fa2c4da74402494554d"
AUX = ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "tokenizer.model",
       "added_tokens.json", "preprocessor_config.json", "processor_config.json"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--temperature", type=float, default=None,
                    help="write this as the model's default sampling temperature (0 = greedy)")
    args = ap.parse_args()

    os.makedirs(args.dst, exist_ok=True)
    model = AutoModelForCausalLM.from_pretrained(args.src, dtype=torch.bfloat16)
    model.save_pretrained(args.dst, safe_serialization=True)
    for f in AUX:
        p = os.path.join(BASE, f)
        if os.path.exists(p):
            shutil.copy(p, os.path.join(args.dst, f))

    gc = json.load(open(os.path.join(BASE, "generation_config.json")))
    if args.temperature is not None:
        gc["temperature"] = args.temperature
        if args.temperature == 0.0:
            gc["do_sample"] = False
            gc.pop("top_k", None)
            gc.pop("top_p", None)
    json.dump(gc, open(os.path.join(args.dst, "generation_config.json"), "w"), indent=2)
    print("packaged", args.dst, "->", sorted(os.listdir(args.dst)))


if __name__ == "__main__":
    main()
