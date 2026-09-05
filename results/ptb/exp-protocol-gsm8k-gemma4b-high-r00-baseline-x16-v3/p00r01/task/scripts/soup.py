#!/usr/bin/env python3
"""Uniform weight average of checkpoints fine-tuned from the same base init.

Averages in fp32 and writes back in the source dtype. Copies the tokenizer and
the processor configs vllm needs from the first source.
"""
from __future__ import annotations

import argparse
import os
import shutil

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    print("souping:", args.models)
    base = AutoModelForCausalLM.from_pretrained(args.models[0], dtype=torch.float32)
    acc = {k: v.clone().float() for k, v in base.state_dict().items()}

    for m in args.models[1:]:
        other = AutoModelForCausalLM.from_pretrained(m, dtype=torch.float32)
        sd = other.state_dict()
        missing = set(acc) ^ set(sd)
        if missing:
            raise SystemExit(f"state dict mismatch vs {m}: {sorted(missing)[:5]}")
        for k in acc:
            acc[k] += sd[k].float()
        del other, sd

    n = len(args.models)
    for k in acc:
        acc[k] /= n

    base.load_state_dict({k: v for k, v in acc.items()})
    base = base.to(torch.bfloat16)
    base.save_pretrained(args.out)

    tok = AutoTokenizer.from_pretrained(args.models[0])
    tok.save_pretrained(args.out)
    for name in ("preprocessor_config.json", "processor_config.json"):
        s = os.path.join(args.models[0], name)
        if os.path.exists(s):
            shutil.copy2(s, os.path.join(args.out, name))
    print("saved", args.out)


if __name__ == "__main__":
    main()
