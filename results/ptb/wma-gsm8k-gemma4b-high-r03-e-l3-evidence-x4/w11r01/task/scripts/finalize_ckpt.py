#!/usr/bin/env python3
"""Make a Trainer checkpoint dir loadable by the grader's vLLM.

Trainer's intermediate checkpoint-N dirs hold weights and config.json only. The
grader loads a gemma3 path as a multimodal model, so the tokenizer and the two
processor configs have to sit next to the weights, and generation_config.json
has to still say eos_token_id [1, 106] -- that list is what stops vLLM at
<end_of_turn> (pitfalls.yaml eos_mismatch, final_model_not_loadable).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil

SNAPSHOT = "/home/ben/hf_cache/hub/models--google--gemma-3-4b-pt/snapshots/cc012e0a6d0787b4adcc0fa2c4da74402494554d"
NEEDED = (
    "tokenizer.json", "tokenizer.model", "tokenizer_config.json",
    "special_tokens_map.json", "added_tokens.json",
    "preprocessor_config.json", "processor_config.json",
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--greedy", action="store_true",
                    help="also write a greedy generation_config.json")
    args = ap.parse_args()

    added = []
    for f in NEEDED:
        src, dst = os.path.join(SNAPSHOT, f), os.path.join(args.ckpt, f)
        if os.path.exists(src) and not os.path.exists(dst):
            shutil.copy(src, dst)
            added.append(f)

    gpath = os.path.join(args.ckpt, "generation_config.json")
    base = json.load(open(os.path.join(SNAPSHOT, "generation_config.json")))
    gen = json.load(open(gpath)) if os.path.exists(gpath) else {}
    if gen.get("eos_token_id") != base["eos_token_id"]:
        print(f"  ! eos_token_id was {gen.get('eos_token_id')}, restoring {base['eos_token_id']}")
    gen = dict(base) | {k: v for k, v in gen.items() if k != "eos_token_id"}
    gen["eos_token_id"] = base["eos_token_id"]
    if args.greedy:
        gen.update({"do_sample": False, "temperature": 0.0, "top_p": 1.0, "top_k": 0})
    json.dump(gen, open(gpath, "w"), indent=2)

    cfg = json.load(open(os.path.join(args.ckpt, "config.json")))
    print(f"{args.ckpt}: added {added or 'nothing'}")
    print(f"  architectures={cfg['architectures']} eos_token_id={gen['eos_token_id']} "
          f"do_sample={gen.get('do_sample')} temperature={gen.get('temperature')}")


if __name__ == "__main__":
    main()
