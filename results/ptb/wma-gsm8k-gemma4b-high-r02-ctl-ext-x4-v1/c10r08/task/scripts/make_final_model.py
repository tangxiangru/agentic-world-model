#!/usr/bin/env python3
"""Copy a checkpoint into final_model/ and verify the grader can load it.

Guards the final_model_not_loadable pitfall: the grader loads final_model/ with
vLLM from a fresh process, using templates/gemma3.jinja and whatever
generation_config.json we leave behind.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys

BASE = ("/home/ben/hf_cache/hub/models--google--gemma-3-4b-pt/snapshots/"
        "cc012e0a6d0787b4adcc0fa2c4da74402494554d")
NEEDED = ["config.json", "generation_config.json", "tokenizer.json",
          "tokenizer_config.json", "special_tokens_map.json",
          "preprocessor_config.json", "processor_config.json",
          "model.safetensors.index.json"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", default="/home/ben/task/final_model")
    ap.add_argument("--greedy", action="store_true",
                    help="write a generation_config that makes vLLM decode greedily")
    ap.add_argument("--verify-only", action="store_true")
    args = ap.parse_args()

    if not args.verify_only:
        if os.path.exists(args.dst):
            print(f"replacing existing {args.dst}", flush=True)
            shutil.rmtree(args.dst)
        shutil.copytree(args.src, args.dst)
        for fn in NEEDED:
            dst = os.path.join(args.dst, fn)
            if not os.path.exists(dst) and os.path.exists(os.path.join(BASE, fn)):
                shutil.copy(os.path.join(BASE, fn), dst)
                print(f"backfilled {fn} from the base snapshot", flush=True)
        if args.greedy:
            gen = {"bos_token_id": 2, "cache_implementation": "hybrid",
                   "do_sample": False, "temperature": 0.0,
                   "eos_token_id": [1, 106], "pad_token_id": 0,
                   "transformers_version": "4.50.0.dev0"}
            with open(os.path.join(args.dst, "generation_config.json"), "w") as f:
                json.dump(gen, f, indent=2)
            print("wrote greedy generation_config.json", flush=True)
        # trainer state / optimizer leftovers are not needed by the grader
        for junk in ("optimizer.pt", "scheduler.pt", "rng_state.pth", "trainer_state.json",
                     "training_args.bin"):
            p = os.path.join(args.dst, junk)
            if os.path.exists(p):
                os.remove(p)

    # ---- verification ------------------------------------------------------
    missing = [fn for fn in NEEDED if not os.path.exists(os.path.join(args.dst, fn))]
    if missing:
        print("MISSING FILES:", missing, flush=True)
        sys.exit(1)
    cfg = json.load(open(os.path.join(args.dst, "config.json")))
    gen = json.load(open(os.path.join(args.dst, "generation_config.json")))
    print("architectures:", cfg["architectures"], flush=True)
    print("generation_config:", gen, flush=True)
    assert cfg["architectures"] == ["Gemma3ForConditionalGeneration"], cfg["architectures"]
    assert 106 in (gen["eos_token_id"] if isinstance(gen["eos_token_id"], list)
                   else [gen["eos_token_id"]]), "vLLM must stop on <end_of_turn>"

    # evaluate.py's model_type(): 'gemma' must be reachable from the path or config
    arch = cfg["architectures"][0].lower()
    assert "gemma" in arch or "gemma" in args.dst.lower(), "evaluate.py cannot pick a template"

    import torch
    from transformers import AutoTokenizer, Gemma3ForConditionalGeneration
    tok = AutoTokenizer.from_pretrained(args.dst)
    assert tok.convert_tokens_to_ids("<end_of_turn>") == 106
    m = Gemma3ForConditionalGeneration.from_pretrained(args.dst, dtype=torch.bfloat16,
                                                       device_map="cpu")
    n = sum(p.numel() for p in m.parameters())
    print(f"loaded on CPU: {n / 1e9:.2f}B params", flush=True)
    print("OK", flush=True)


if __name__ == "__main__":
    main()
