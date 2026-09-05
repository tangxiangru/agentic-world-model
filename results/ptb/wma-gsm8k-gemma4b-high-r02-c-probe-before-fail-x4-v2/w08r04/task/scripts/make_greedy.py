#!/usr/bin/env python3
"""Materialise a checkpoint with a greedy generation_config.

evaluate.py passes no temperature to inspect_ai, and vLLM 0.11 logs
"Default sampling parameters have been overridden by the model's Hugging Face
generation config": whatever is in the checkpoint's generation_config.json IS the
decode config at grading time. gemma-3-4b-pt ships do_sample/top_k 64/top_p 0.95
with no temperature, so the default read is T=1.0 sampling.

Weights are hard-linked (no copy), only the json files are rewritten.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--temperature", type=float, default=0.0)
    args = ap.parse_args()

    os.makedirs(args.dst, exist_ok=True)
    for f in os.listdir(args.src):
        s, d = os.path.join(args.src, f), os.path.join(args.dst, f)
        if os.path.isdir(s) or os.path.exists(d):
            continue
        try:
            os.link(s, d)
        except OSError:
            shutil.copy(s, d)

    gc_path = os.path.join(args.dst, "generation_config.json")
    gc = json.load(open(gc_path)) if os.path.exists(gc_path) else {}
    gc.update({"temperature": args.temperature, "top_p": 1.0, "top_k": 0,
               "do_sample": args.temperature > 0})
    eos = gc.get("eos_token_id")
    assert eos == [1, 106] or (isinstance(eos, list) and 106 in eos), \
        f"eos_token_id must still contain 106 (<end_of_turn>), got {eos!r}"
    os.remove(gc_path)
    json.dump(gc, open(gc_path, "w"), indent=2)
    print(json.dumps(gc, indent=2))
    print("wrote", args.dst)


if __name__ == "__main__":
    main()
