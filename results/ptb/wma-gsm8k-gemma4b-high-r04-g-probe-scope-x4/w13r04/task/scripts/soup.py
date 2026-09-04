#!/usr/bin/env python3
"""Weight-space average of two or more checkpoints of the same architecture."""
from __future__ import annotations

import argparse
import json
import os
import shutil

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

GEN = {
    "bos_token_id": 2,
    "eos_token_id": [1, 106],
    "pad_token_id": 0,
    "do_sample": False,
    "temperature": 0.0,
    "top_p": 1.0,
    "top_k": 0,
    "transformers_version": "4.57.3",
}
COPY = (
    "preprocessor_config.json",
    "processor_config.json",
    "added_tokens.json",
    "special_tokens_map.json",
    "tokenizer.model",
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--base", default=os.environ["PTB_BASE_MODEL_SNAPSHOT"])
    ap.add_argument("ckpts", nargs="+", help="path[:weight]")
    args = ap.parse_args()

    paths, weights = [], []
    for spec in args.ckpts:
        p, _, w = spec.rpartition(":")
        if not p or os.path.exists(spec):
            p, w = spec, ""
        paths.append(p)
        weights.append(float(w) if w else 1.0)
    tot = sum(weights)
    weights = [w / tot for w in weights]
    print("soup:", list(zip(paths, weights)), flush=True)

    model = AutoModelForCausalLM.from_pretrained(paths[0], dtype=torch.float32)
    acc = {k: v.clone() * weights[0] for k, v in model.state_dict().items()}
    for p, w in zip(paths[1:], weights[1:]):
        other = AutoModelForCausalLM.from_pretrained(p, dtype=torch.float32)
        sd = other.state_dict()
        assert set(sd) == set(acc), "state dict mismatch"
        for k in acc:
            acc[k] += sd[k] * w
        del other, sd
    model.load_state_dict(acc)
    model = model.to(torch.bfloat16)
    model.config.use_cache = True
    model.generation_config = GenerationConfig(
        bos_token_id=2, eos_token_id=[1, 106], pad_token_id=0
    )
    model.save_pretrained(args.out, safe_serialization=True)
    AutoTokenizer.from_pretrained(args.base).save_pretrained(args.out)
    for fn in COPY:
        src = os.path.join(args.base, fn)
        dst = os.path.join(args.out, fn)
        if os.path.exists(src) and not os.path.exists(dst):
            shutil.copy(src, dst)
    with open(os.path.join(args.out, "generation_config.json"), "w") as f:
        json.dump(GEN, f, indent=2)
    print("wrote", args.out, sorted(os.listdir(args.out)), flush=True)


if __name__ == "__main__":
    main()
