#!/usr/bin/env python3
"""Copy a checkpoint into final_model/ and verify the grader can load it.

Guards pitfall final_model_not_loadable: the directory must carry the full weights
(not an adapter), the tokenizer, and the base generation_config whose eos list
contains <end_of_turn> (106) - that is what stops vLLM at the end of a turn.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil

os.environ.setdefault("HF_HOME", "/home/ben/hf_cache")

SNAPSHOT = (
    "/home/ben/hf_cache/hub/models--google--gemma-3-4b-pt/snapshots/"
    "cc012e0a6d0787b4adcc0fa2c4da74402494554d"
)
NEEDED = ("generation_config.json", "preprocessor_config.json", "processor_config.json")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", default="/home/ben/task/final_model")
    args = ap.parse_args()

    if os.path.exists(args.dst):
        shutil.rmtree(args.dst)
    shutil.copytree(args.src, args.dst)

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(SNAPSHOT)
    tok.save_pretrained(args.dst)
    for name in NEEDED:
        src = os.path.join(SNAPSHOT, name)
        if os.path.exists(src) and not os.path.exists(os.path.join(args.dst, name)):
            shutil.copy(src, os.path.join(args.dst, name))
    for junk in ("trainer_state.json", "training_args.bin", "optimizer.pt", "scheduler.pt", "rng_state.pth"):
        p = os.path.join(args.dst, junk)
        if os.path.exists(p):
            os.remove(p)

    cfg = json.load(open(os.path.join(args.dst, "config.json")))
    gen = json.load(open(os.path.join(args.dst, "generation_config.json")))
    print("architectures:", cfg["architectures"])
    print("eos_token_id:", gen.get("eos_token_id"))
    assert 106 in (gen.get("eos_token_id") or []), "generation_config lost <end_of_turn>"

    # evaluate.py routes on config.json when the path has no model name in it
    arch = cfg["architectures"][0].lower()
    assert "gemma" in arch, f"evaluate.py would not pick gemma3.jinja for {arch}"

    import torch
    from transformers import Gemma3ForConditionalGeneration

    m = Gemma3ForConditionalGeneration.from_pretrained(args.dst, dtype=torch.bfloat16)
    n = sum(p.numel() for p in m.parameters())
    print(f"loaded {args.dst} on CPU with transformers: {n / 1e9:.2f}B params")
    print("files:", sorted(os.listdir(args.dst)))


if __name__ == "__main__":
    main()
