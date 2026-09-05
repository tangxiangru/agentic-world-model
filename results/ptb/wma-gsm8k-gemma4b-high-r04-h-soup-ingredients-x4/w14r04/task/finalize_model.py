#!/usr/bin/env python3
"""Make a trained checkpoint loadable by the grader's fresh vLLM process.

Pitfall final_model_not_loadable: the grader runs `evaluate.py --model-path
final_model` from a new process, resolves model_type from config.json
architectures, hands templates/gemma3.jinja to vLLM, and reads generation_config
for its decode defaults.  This script copies the tokenizer/processor files from
the immutable base snapshot, pins a greedy generation_config that stops on both
<eos> (1) and <end_of_turn> (106), and then loads the result once to prove it.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil

SNAP = os.environ.get(
    "PTB_BASE_MODEL_SNAPSHOT",
    "/home/ben/hf_cache/hub/models--google--gemma-3-4b-pt/snapshots/cc012e0a6d0787b4adcc0fa2c4da74402494554d",
)

COPY = [
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "preprocessor_config.json",
    "processor_config.json",
]

GEN_CFG = {
    "bos_token_id": 2,
    "eos_token_id": [1, 106],
    "pad_token_id": 0,
    "cache_implementation": "hybrid",
    "do_sample": False,
    "temperature": 0.0,
    "top_p": 1.0,
    "top_k": -1,
    "transformers_version": "4.57.3",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="trainer output dir with the weights")
    ap.add_argument("--dst", required=True)
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.dst, exist_ok=True)
    for f in os.listdir(args.src):
        if f.startswith(("optimizer", "scheduler", "rng_state", "trainer_state", "training_args")):
            continue
        s = os.path.join(args.src, f)
        if os.path.isfile(s):
            shutil.copy2(s, os.path.join(args.dst, f))
    for f in COPY:
        s = os.path.join(SNAP, f)
        if os.path.exists(s):
            shutil.copy2(s, os.path.join(args.dst, f))

    with open(os.path.join(args.dst, "generation_config.json"), "w") as f:
        json.dump(GEN_CFG, f, indent=2)

    cfg = json.load(open(os.path.join(args.dst, "config.json")))
    arch = cfg["architectures"][0]
    print("architectures:", arch)
    assert "gemma" in arch.lower(), arch
    print("files:", sorted(os.listdir(args.dst)))

    if args.verify:
        import torch
        from transformers import AutoTokenizer, Gemma3ForConditionalGeneration

        tok = AutoTokenizer.from_pretrained(args.dst)
        assert tok.convert_tokens_to_ids("<end_of_turn>") == 106
        m = Gemma3ForConditionalGeneration.from_pretrained(
            args.dst, torch_dtype=torch.bfloat16, device_map="cpu"
        )
        n = sum(p.numel() for p in m.parameters())
        print("loaded on CPU ok, params:", round(n / 1e9, 3), "B")


if __name__ == "__main__":
    main()
