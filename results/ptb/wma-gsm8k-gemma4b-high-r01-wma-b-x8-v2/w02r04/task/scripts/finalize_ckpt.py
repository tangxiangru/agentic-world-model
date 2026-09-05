#!/usr/bin/env python3
"""Make a trained checkpoint loadable by the grader's vLLM, byte-checked.

Two things the Trainer does not save and vLLM 0.11.0 needs for gemma-3:
  * preprocessor_config.json / processor_config.json - Gemma3Processor is built
    at engine start even for a text-only request; without them the engine
    raises before generating a token.
  * a generation_config.json whose eos_token_id still contains 106
    (<end_of_turn>); if it collapses to 1 the model never terminates.

--decode greedy additionally rewrites generation_config.json so vLLM's
`--generation-config auto` default resolves to temperature 0 instead of the
base model's do_sample/top_k 64/top_p 0.95.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil

BASE = os.environ.get(
    "PTB_BASE_MODEL_SNAPSHOT",
    "/home/ben/hf_cache/hub/models--google--gemma-3-4b-pt/snapshots/"
    "cc012e0a6d0787b4adcc0fa2c4da74402494554d",
)

AUX = [
    "preprocessor_config.json",
    "processor_config.json",
    "added_tokens.json",
    "special_tokens_map.json",
    "tokenizer_config.json",
    "tokenizer.json",
    "tokenizer.model",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--base", default=BASE)
    ap.add_argument("--decode", choices=["base", "greedy"], default="base")
    args = ap.parse_args()

    for name in AUX:
        src = os.path.join(args.base, name)
        dst = os.path.join(args.ckpt, name)
        if os.path.exists(src) and not os.path.exists(dst):
            shutil.copyfile(src, dst)
            print("copied", name)

    base_gc = json.load(open(os.path.join(args.base, "generation_config.json")))
    gc_path = os.path.join(args.ckpt, "generation_config.json")
    gc = json.load(open(gc_path)) if os.path.exists(gc_path) else {}

    if args.decode == "base":
        gc = dict(base_gc)
    else:
        gc = {
            "bos_token_id": base_gc["bos_token_id"],
            "eos_token_id": base_gc["eos_token_id"],
            "pad_token_id": base_gc["pad_token_id"],
            "cache_implementation": base_gc.get("cache_implementation", "hybrid"),
            "do_sample": False,
            "temperature": 0.0,
        }
    json.dump(gc, open(gc_path, "w"), indent=2)
    print("generation_config:", json.dumps(gc))

    eos = gc["eos_token_id"]
    assert isinstance(eos, list) and 106 in eos, f"eos_token_id lost 106: {eos}"

    cfg = json.load(open(os.path.join(args.ckpt, "config.json")))
    print("architectures:", cfg["architectures"])
    missing = [n for n in ("preprocessor_config.json", "tokenizer.json")
               if not os.path.exists(os.path.join(args.ckpt, n))]
    assert not missing, f"still missing {missing}"
    print("OK", args.ckpt)


if __name__ == "__main__":
    main()
