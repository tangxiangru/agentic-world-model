#!/usr/bin/env python3
"""Copy a checkpoint into final_model/ with the chosen decode defaults, then verify it.

Verification mirrors the grader's own load path as closely as possible without a
GPU: config.json names a gemma architecture (so evaluate.py picks gemma3.jinja),
the tokenizer loads, every shard in the index is present and readable, and the
generation_config stops on the same token the grading template stops on.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil

GREEDY = {
    "bos_token_id": 2,
    "cache_implementation": "hybrid",
    "do_sample": False,
    "eos_token_id": [1, 106],
    "pad_token_id": 0,
    "temperature": 0.0,
    "top_k": 1,
    "top_p": 1.0,
    "transformers_version": "4.57.3",
}
SAMPLED = {
    "bos_token_id": 2,
    "cache_implementation": "hybrid",
    "do_sample": True,
    "eos_token_id": [1, 106],
    "pad_token_id": 0,
    "top_k": 64,
    "top_p": 0.95,
    "transformers_version": "4.57.3",
}
SKIP = {"training_args.bin", "trainer_state.json", "optimizer.pt", "scheduler.pt",
        "rng_state.pth", "generation_config.json"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", default="/home/ben/task/final_model")
    ap.add_argument("--decode", choices=["greedy", "sampled"], default="greedy")
    args = ap.parse_args()

    src = os.path.realpath(args.src)
    if os.path.exists(args.dst):
        shutil.rmtree(args.dst)
    os.makedirs(args.dst)
    for fn in sorted(os.listdir(src)):
        if fn in SKIP or fn.startswith("."):
            continue
        s = os.path.realpath(os.path.join(src, fn))
        if os.path.isdir(s):
            continue
        shutil.copy(s, os.path.join(args.dst, fn))
        print("copied", fn, os.path.getsize(s))
    gc = GREEDY if args.decode == "greedy" else SAMPLED
    with open(os.path.join(args.dst, "generation_config.json"), "w") as f:
        json.dump(gc, f, indent=2)

    # ---- verification ----
    cfg = json.load(open(os.path.join(args.dst, "config.json")))
    arch = cfg["architectures"][0]
    assert "gemma" in arch.lower(), arch
    idx = json.load(open(os.path.join(args.dst, "model.safetensors.index.json")))
    shards = sorted(set(idx["weight_map"].values()))
    for sh in shards:
        p = os.path.join(args.dst, sh)
        assert os.path.getsize(p) > 10**6, p
    from safetensors import safe_open

    n_t = 0
    for sh in shards:
        with safe_open(os.path.join(args.dst, sh), framework="pt") as f:
            n_t += len(f.keys())
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.dst)
    assert tok.convert_ids_to_tokens(106) == "<end_of_turn>"
    gcj = json.load(open(os.path.join(args.dst, "generation_config.json")))
    assert 106 in gcj["eos_token_id"]
    print(json.dumps({
        "dst": args.dst, "architecture": arch, "shards": shards,
        "tensors": n_t, "decode": args.decode,
        "eos_token_id": gcj["eos_token_id"],
        "files": sorted(os.listdir(args.dst)),
    }, indent=2))


if __name__ == "__main__":
    main()
