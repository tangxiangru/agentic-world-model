#!/usr/bin/env python3
"""Uniform weight average ("soup") of two or more checkpoints of the same
architecture, saved as a directory the grader's vLLM can load.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil

import torch
from transformers import AutoTokenizer, Gemma3ForConditionalGeneration

SNAP = os.environ.get(
    "PTB_BASE_MODEL_SNAPSHOT",
    "/home/ben/hf_cache/hub/models--google--gemma-3-4b-pt/snapshots/"
    "cc012e0a6d0787b4adcc0fa2c4da74402494554d",
)

GREEDY = {
    "bos_token_id": 2,
    "cache_implementation": "hybrid",
    "do_sample": True,
    "eos_token_id": [1, 106],
    "pad_token_id": 0,
    "temperature": 0.0,
    "top_k": 0,
    "top_p": 1.0,
    "transformers_version": "4.50.0.dev0",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--weights", nargs="*", type=float, default=None)
    args = ap.parse_args()

    w = args.weights or [1.0 / len(args.inputs)] * len(args.inputs)
    assert len(w) == len(args.inputs)
    total = sum(w)
    w = [x / total for x in w]
    print(f"[soup] weights {w}")

    model = Gemma3ForConditionalGeneration.from_pretrained(
        args.inputs[0], dtype=torch.float32, device_map="cpu"
    )
    acc = {k: v.detach().clone() * w[0] for k, v in model.state_dict().items()}
    for path, wi in zip(args.inputs[1:], w[1:]):
        other = Gemma3ForConditionalGeneration.from_pretrained(
            path, dtype=torch.float32, device_map="cpu"
        )
        sd = other.state_dict()
        for k in acc:
            acc[k] += sd[k] * wi
        del other, sd
        print(f"[soup] merged {path}")
    for k in acc:
        acc[k] = acc[k].to(torch.bfloat16)
    model.load_state_dict(acc)
    model = model.to(torch.bfloat16)
    model.generation_config.do_sample = True
    model.generation_config.temperature = 1.0
    model.generation_config.top_k = 64
    model.generation_config.top_p = 0.95
    model.save_pretrained(args.out)
    AutoTokenizer.from_pretrained(SNAP).save_pretrained(args.out)
    for fn in ("preprocessor_config.json", "processor_config.json"):
        src = os.path.join(SNAP, fn)
        if os.path.exists(src):
            shutil.copyfile(src, os.path.join(args.out, fn))
    with open(os.path.join(args.out, "generation_config.json"), "w") as f:
        json.dump(GREEDY, f, indent=2)
    print(f"[soup] wrote {args.out}")


if __name__ == "__main__":
    main()
