#!/usr/bin/env python3
"""Copy a trained checkpoint into ./final_model with a greedy generation config."""
from __future__ import annotations

import argparse
import json
import os
import shutil

GREEDY_GEN_CONFIG = {
    "bos_token_id": 2,
    "cache_implementation": "hybrid",
    "do_sample": False,
    "eos_token_id": [1, 106],
    "pad_token_id": 0,
    "temperature": 0.0,
    "top_k": 1,
    "top_p": 1.0,
    "transformers_version": "4.50.0.dev0",
}

SKIP_PREFIX = ("checkpoint-",)
SKIP_FILES = {"training_args.bin", "optimizer.pt", "scheduler.pt", "rng_state.pth", "trainer_state.json"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", default="final_model")
    ap.add_argument("--sampling", action="store_true", help="keep stock sampling config")
    args = ap.parse_args()

    if os.path.exists(args.dst):
        shutil.rmtree(args.dst)
    os.makedirs(args.dst)
    for name in sorted(os.listdir(args.src)):
        if name.startswith(SKIP_PREFIX) or name in SKIP_FILES:
            continue
        src = os.path.join(args.src, name)
        if os.path.isdir(src):
            continue
        shutil.copy(os.path.realpath(src), os.path.join(args.dst, name))
    if not args.sampling:
        with open(os.path.join(args.dst, "generation_config.json"), "w") as f:
            json.dump(GREEDY_GEN_CONFIG, f, indent=2)
    print("wrote", args.dst)
    print(sorted(os.listdir(args.dst)))


if __name__ == "__main__":
    main()
