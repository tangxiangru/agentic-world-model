#!/usr/bin/env python3
"""Turn a Trainer checkpoint (fp32 weights, no tokenizer) into a directory the
grader's vLLM can load: bf16 weights + tokenizer + processor + a greedy
generation_config.json whose eos_token_id still contains <end_of_turn> (106).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor, AutoTokenizer

BASE = "/home/ben/hf_cache/hub/models--google--gemma-3-4b-pt/snapshots/cc012e0a6d0787b4adcc0fa2c4da74402494554d"

# vLLM reads temperature/top_p/top_k out of this file (generation_config="auto")
# and inspect_ai sends no temperature, so this is what decides the decode.
# temperature 0.0 -> exact greedy in vLLM (< _SAMPLING_EPS); top_k is omitted
# rather than set to the -1 sentinel.
GREEDY = {
    "bos_token_id": 2,
    "cache_implementation": "hybrid",
    "do_sample": False,
    "temperature": 0.0,
    "top_p": 1.0,
    "eos_token_id": [1, 106],
    "pad_token_id": 0,
    "transformers_version": "4.57.3",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--base", default=BASE)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    m = AutoModelForImageTextToText.from_pretrained(args.ckpt, torch_dtype=torch.bfloat16)
    m.config.use_cache = True
    m.save_pretrained(args.out, safe_serialization=True)
    AutoTokenizer.from_pretrained(args.base).save_pretrained(args.out)
    try:
        AutoProcessor.from_pretrained(args.base).save_pretrained(args.out)
    except Exception as e:
        print("processor save failed:", e)
    for fn in ("preprocessor_config.json", "processor_config.json"):
        src = os.path.join(args.base, fn)
        if os.path.exists(src) and not os.path.exists(os.path.join(args.out, fn)):
            shutil.copy(src, os.path.join(args.out, fn))
    json.dump(GREEDY, open(os.path.join(args.out, "generation_config.json"), "w"), indent=2)
    print("packaged", args.ckpt, "->", args.out)
    print(sorted(os.listdir(args.out)))


if __name__ == "__main__":
    main()
